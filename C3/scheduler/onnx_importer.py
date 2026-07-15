from __future__ import annotations

from pathlib import Path
from typing import Any

import onnx
from onnx import AttributeProto, TensorProto, numpy_helper, shape_inference

from .graph import Graph, GraphNode
from .types import ExternalTensorReference, TensorInfo


def _dtype_name(elem_type: int) -> str:
    """Convert an ONNX TensorProto dtype enum to a stable string."""
    try:
        return TensorProto.DataType.Name(elem_type)
    except ValueError:
        return f"UNKNOWN_{elem_type}"


def _parse_shape(
    value_info: onnx.ValueInfoProto,
) -> tuple[int | str | None, ...]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return ()

    dims: list[int | str | None] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param") and dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return tuple(dims)


def _tensor_info_from_value_info(
    value_info: onnx.ValueInfoProto,
) -> TensorInfo:
    tensor_type = value_info.type.tensor_type
    elem_type = int(tensor_type.elem_type) if tensor_type.elem_type else 0
    return TensorInfo(
        name=value_info.name,
        dtype=_dtype_name(elem_type),
        shape=_parse_shape(value_info),
    )


def _attribute_to_python(attribute: AttributeProto) -> Any:
    """Convert a common ONNX attribute into a plain Python value."""
    attr_type = attribute.type

    if attr_type == AttributeProto.FLOAT:
        return float(attribute.f)
    if attr_type == AttributeProto.INT:
        return int(attribute.i)
    if attr_type == AttributeProto.STRING:
        return attribute.s.decode("utf-8", errors="replace")
    if attr_type == AttributeProto.FLOATS:
        return tuple(float(value) for value in attribute.floats)
    if attr_type == AttributeProto.INTS:
        return tuple(int(value) for value in attribute.ints)
    if attr_type == AttributeProto.STRINGS:
        return tuple(
            value.decode("utf-8", errors="replace")
            for value in attribute.strings
        )
    if attr_type == AttributeProto.TENSOR:
        return numpy_helper.to_array(attribute.t)
    if attr_type == AttributeProto.TENSORS:
        return tuple(
            numpy_helper.to_array(tensor)
            for tensor in attribute.tensors
        )
    if attr_type == AttributeProto.GRAPH:
        return {
            "name": attribute.g.name,
            "node_count": len(attribute.g.node),
        }
    if attr_type == AttributeProto.GRAPHS:
        return tuple(
            {"name": graph.name, "node_count": len(graph.node)}
            for graph in attribute.graphs
        )
    if attr_type == AttributeProto.SPARSE_TENSOR:
        return "<sparse_tensor>"
    if attr_type == AttributeProto.SPARSE_TENSORS:
        return tuple("<sparse_tensor>" for _ in attribute.sparse_tensors)
    if attr_type == AttributeProto.TYPE_PROTO:
        return str(attribute.tp)
    if attr_type == AttributeProto.TYPE_PROTOS:
        return tuple(str(value) for value in attribute.type_protos)
    return None


def _collect_tensor_info(
    graph_proto: onnx.GraphProto,
) -> dict[str, TensorInfo]:
    tensors: dict[str, TensorInfo] = {}

    for value_info in (
        list(graph_proto.input)
        + list(graph_proto.output)
        + list(graph_proto.value_info)
    ):
        if value_info.name:
            tensors[value_info.name] = _tensor_info_from_value_info(
                value_info
            )

    for initializer in graph_proto.initializer:
        tensors[initializer.name] = TensorInfo(
            name=initializer.name,
            dtype=_dtype_name(initializer.data_type),
            shape=tuple(int(dim) for dim in initializer.dims),
        )
    return tensors


def _unique_node_name(
    original_name: str,
    op_type: str,
    index: int,
    used_names: set[str],
) -> str:
    base = original_name or f"{op_type}_{index}"
    candidate = base
    suffix = 1
    while candidate in used_names:
        candidate = f"{base}__{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _initializer_value(
    initializer: onnx.TensorProto,
) -> Any:
    metadata = {
        item.key: item.value
        for item in initializer.external_data
    }
    if not metadata:
        return numpy_helper.to_array(initializer)

    return ExternalTensorReference(
        name=initializer.name,
        dtype=_dtype_name(initializer.data_type),
        shape=tuple(int(dim) for dim in initializer.dims),
        location=metadata["location"],
        offset=int(metadata.get("offset", "0")),
        length=(
            int(metadata["length"])
            if "length" in metadata
            else None
        ),
    )


def import_onnx_graph(model_path: str | Path) -> Graph:
    """Convert ONNX into the internal graph without materializing external weights."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    # Passing the path validates external-data models without serializing them.
    onnx.checker.check_model(str(path))
    model = onnx.load(str(path), load_external_data=False)

    try:
        inferred_model = shape_inference.infer_shapes(
            model,
            check_type=True,
            strict_mode=False,
            data_prop=False,
        )
    except Exception:
        inferred_model = model

    graph_proto = inferred_model.graph
    initializer_names = {
        initializer.name
        for initializer in graph_proto.initializer
    }

    graph_inputs = [
        value_info.name
        for value_info in graph_proto.input
        if value_info.name and value_info.name not in initializer_names
    ]
    graph_outputs = [
        value_info.name
        for value_info in graph_proto.output
        if value_info.name
    ]
    initializers = {
        initializer.name: _initializer_value(initializer)
        for initializer in graph_proto.initializer
    }
    tensors = _collect_tensor_info(graph_proto)

    used_names: set[str] = set()
    nodes: list[GraphNode] = []
    for index, node_proto in enumerate(graph_proto.node):
        name = _unique_node_name(
            node_proto.name,
            node_proto.op_type,
            index,
            used_names,
        )
        attributes = {
            attribute.name: _attribute_to_python(attribute)
            for attribute in node_proto.attribute
        }
        nodes.append(
            GraphNode(
                name=name,
                op_type=node_proto.op_type,
                inputs=[
                    input_name
                    for input_name in node_proto.input
                    if input_name
                ],
                outputs=[
                    output_name
                    for output_name in node_proto.output
                    if output_name
                ],
                attributes=attributes,
            )
        )

    graph = Graph(
        nodes=nodes,
        inputs=graph_inputs,
        outputs=graph_outputs,
        initializers=initializers,
        tensors=tensors,
    )
    graph.validate()
    return graph
