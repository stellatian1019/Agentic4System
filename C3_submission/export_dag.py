#!/usr/bin/env python3
"""
C3.1: ONNX -> DAG JSON exporter.

Usage:
    python3 export_dag.py --onnx model.onnx --output dag.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set

import onnx
from onnx import TensorProto, ValueInfoProto


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an ONNX computation graph as DAG JSON."
    )
    parser.add_argument(
        "--onnx",
        required=True,
        type=Path,
        help="Path to the input ONNX model.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to the output DAG JSON file.",
    )
    return parser.parse_args()


def dtype_to_string(elem_type: int) -> str:
    """
    Convert an ONNX TensorProto element type to a stable readable string.

    Examples:
        TensorProto.FLOAT -> "FLOAT"
        TensorProto.INT64 -> "INT64"
    """
    try:
        return TensorProto.DataType.Name(elem_type)
    except ValueError:
        return f"UNKNOWN_{elem_type}"


def dim_to_json(dim: Any) -> Any:
    """
    Convert an ONNX TensorShapeProto.Dimension to JSON.

    - Concrete dimension: integer
    - Symbolic dimension: string
    - Unknown dimension: "?"
    """
    if dim.HasField("dim_value"):
        return int(dim.dim_value)

    if dim.HasField("dim_param") and dim.dim_param:
        return str(dim.dim_param)

    return "?"


def value_info_to_json(value_info: ValueInfoProto) -> Dict[str, Any]:
    """
    Convert ONNX ValueInfoProto to the JSON tensor descriptor required by C3.1.
    """
    tensor_type = value_info.type.tensor_type

    if not value_info.type.HasField("tensor_type"):
        return {
            "name": value_info.name,
            "dtype": "UNKNOWN",
            "shape": [],
        }

    dtype = dtype_to_string(tensor_type.elem_type)

    if tensor_type.HasField("shape"):
        shape = [dim_to_json(dim) for dim in tensor_type.shape.dim]
    else:
        shape = []

    return {
        "name": value_info.name,
        "dtype": dtype,
        "shape": shape,
    }


def make_unique_node_names(nodes: Sequence[onnx.NodeProto]) -> List[str]:
    """
    Return one deterministic, unique name for every ONNX node.

    ONNX allows node.name to be empty or duplicated. The evaluator needs a
    usable DAG, so anonymous or duplicate names are replaced deterministically.
    """
    names: List[str] = []
    used: Set[str] = set()

    for index, node in enumerate(nodes):
        preferred = node.name.strip() if node.name else ""
        base = preferred or f"{node.op_type}_{index}"

        candidate = base
        suffix = 1
        while candidate in used:
            candidate = f"{base}__{suffix}"
            suffix += 1

        used.add(candidate)
        names.append(candidate)

    return names


def build_nodes(
    nodes: Sequence[onnx.NodeProto],
    node_names: Sequence[str],
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []

    for node, unique_name in zip(nodes, node_names):
        result.append(
            {
                "name": unique_name,
                "op_type": node.op_type,
                "inputs": [name for name in node.input if name],
                "outputs": [name for name in node.output if name],
            }
        )

    return result


def build_edges(
    nodes: Sequence[onnx.NodeProto],
    node_names: Sequence[str],
) -> List[Dict[str, str]]:
    """
    Build node-to-node data dependency edges.

    An edge exists when one node produces a tensor consumed by another node.
    Graph inputs and initializers are not represented as source nodes, so they
    do not produce edges.
    """
    producer_of: Dict[str, str] = {}

    for node, node_name in zip(nodes, node_names):
        for tensor_name in node.output:
            if tensor_name:
                producer_of[tensor_name] = node_name

    edges: List[Dict[str, str]] = []
    seen: Set[tuple[str, str, str]] = set()

    for node, dst_name in zip(nodes, node_names):
        for tensor_name in node.input:
            if not tensor_name:
                continue

            src_name = producer_of.get(tensor_name)
            if src_name is None:
                continue

            key = (src_name, dst_name, tensor_name)
            if key in seen:
                continue

            seen.add(key)
            edges.append(
                {
                    "src_node": src_name,
                    "dst_node": dst_name,
                    "tensor": tensor_name,
                }
            )

    return edges


def collect_graph_inputs(model: onnx.ModelProto) -> List[Dict[str, Any]]:
    """
    Return true model inputs, excluding weights/constants stored as initializers.
    """
    initializer_names = {initializer.name for initializer in model.graph.initializer}

    return [
        value_info_to_json(value_info)
        for value_info in model.graph.input
        if value_info.name not in initializer_names
    ]


def collect_graph_outputs(model: onnx.ModelProto) -> List[Dict[str, Any]]:
    return [value_info_to_json(value_info) for value_info in model.graph.output]


def export_dag(model: onnx.ModelProto) -> Dict[str, Any]:
    nodes = list(model.graph.node)
    node_names = make_unique_node_names(nodes)

    return {
        "format_version": "1.0",
        "graph_inputs": collect_graph_inputs(model),
        "graph_outputs": collect_graph_outputs(model),
        "nodes": build_nodes(nodes, node_names),
        "edges": build_edges(nodes, node_names),
    }


def load_and_validate_model(path: Path) -> onnx.ModelProto:
    if not path.exists():
        raise FileNotFoundError(f"ONNX file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"ONNX path is not a regular file: {path}")

    model = onnx.load(str(path), load_external_data=False)
    if not any(item.external_data for item in model.graph.initializer):
        onnx.checker.check_model(model)
    return model

def write_json(data: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    temporary_path.replace(output_path)


def main() -> int:
    args = parse_args()

    try:
        model = load_and_validate_model(args.onnx)
        dag = export_dag(model)
        write_json(dag, args.output)
        return 0
    except Exception as exc:
        print(f"export_dag.py: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
