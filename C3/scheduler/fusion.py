from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .graph import Graph, GraphNode


@dataclass(frozen=True)
class FusionRecord:
    pattern: str
    fused_node: str
    original_nodes: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


class FusionPass:
    """
    Baseline C3.3 fusion pass.

    Supported patterns:
    - Conv + BatchNormalization
    - Conv + Relu
    - Gemm/MatMul + Add
    - Add + Relu
    - LayerNorm subgraph:
      ReduceMean -> Sub -> Mul -> ReduceMean -> Add -> Sqrt -> Div
    """

    name = "Fusion"

    def __init__(self) -> None:
        self._fusion_counter = 0

    def _new_name(self, pattern: str) -> str:
        safe = pattern.replace("+", "_").replace(" ", "_").lower()
        name = f"__c3_fused_{safe}_{self._fusion_counter}"
        self._fusion_counter += 1
        return name

    @staticmethod
    def _single_consumer(graph: Graph, tensor_name: str) -> GraphNode | None:
        consumers = graph.get_consumers(tensor_name)
        return consumers[0] if len(consumers) == 1 else None

    @staticmethod
    def _external_inputs(
        nodes: Iterable[GraphNode],
    ) -> list[str]:
        node_list = list(nodes)
        produced = {
            tensor_name
            for node in node_list
            for tensor_name in node.outputs
        }
        result: list[str] = []
        seen: set[str] = set()

        for node in node_list:
            for tensor_name in node.inputs:
                if not tensor_name or tensor_name in produced:
                    continue
                if tensor_name not in seen:
                    result.append(tensor_name)
                    seen.add(tensor_name)
        return result

    @staticmethod
    def _external_outputs(
        graph: Graph,
        nodes: Iterable[GraphNode],
    ) -> list[str]:
        node_list = list(nodes)
        node_names = {node.name for node in node_list}
        result: list[str] = []
        seen: set[str] = set()

        for node in node_list:
            for tensor_name in node.outputs:
                consumers = graph.get_consumers(tensor_name)
                escapes = (
                    tensor_name in graph.outputs
                    or any(consumer.name not in node_names for consumer in consumers)
                )
                if escapes and tensor_name not in seen:
                    result.append(tensor_name)
                    seen.add(tensor_name)

        # For a straight-line fusion, the last node's outputs are the natural
        # outputs even when there are no explicit external consumers.
        if not result and node_list:
            result.extend(node_list[-1].outputs)
        return result

    def _fuse_nodes(
        self,
        graph: Graph,
        nodes: list[GraphNode],
        *,
        pattern: str,
        fused_op_type: str,
        extra_attributes: dict[str, Any] | None = None,
    ) -> FusionRecord:
        inputs = self._external_inputs(nodes)
        outputs = self._external_outputs(graph, nodes)
        fused_name = self._new_name(pattern)

        attributes: dict[str, Any] = {
            "fusion_pattern": pattern,
            "original_nodes": tuple(node.name for node in nodes),
            "original_op_types": tuple(node.op_type for node in nodes),
        }
        if extra_attributes:
            attributes.update(extra_attributes)

        fused = GraphNode(
            name=fused_name,
            op_type=fused_op_type,
            inputs=inputs,
            outputs=outputs,
            attributes=attributes,
        )
        graph.replace_nodes(nodes, fused)

        return FusionRecord(
            pattern=pattern,
            fused_node=fused_name,
            original_nodes=tuple(node.name for node in nodes),
            inputs=tuple(inputs),
            outputs=tuple(outputs),
        )

    def _find_pair(
        self,
        graph: Graph,
        first_ops: set[str],
        second_ops: set[str],
    ) -> tuple[GraphNode, GraphNode] | None:
        for first in graph.topological_nodes():
            if first.op_type not in first_ops or len(first.outputs) != 1:
                continue

            second = self._single_consumer(graph, first.outputs[0])
            if second is None or second.op_type not in second_ops:
                continue

            # The intermediate tensor must not also be a graph output.
            if first.outputs[0] in graph.outputs:
                continue

            return first, second
        return None

    def _find_layernorm_chain(
        self,
        graph: Graph,
    ) -> list[GraphNode] | None:
        expected = [
            "ReduceMean",
            "Sub",
            "Mul",
            "ReduceMean",
            "Add",
            "Sqrt",
            "Div",
        ]

        for start in graph.topological_nodes():
            if start.op_type != expected[0] or len(start.outputs) != 1:
                continue

            chain = [start]
            current = start
            valid = True

            for expected_op in expected[1:]:
                next_node = self._single_consumer(
                    graph,
                    current.outputs[0],
                )
                if next_node is None or next_node.op_type != expected_op:
                    valid = False
                    break
                if current.outputs[0] in graph.outputs:
                    valid = False
                    break
                chain.append(next_node)
                current = next_node

            if valid:
                return chain

        return None

    def run(self, graph: Graph) -> dict[str, Any]:
        fusion_log: list[dict[str, Any]] = []
        pattern_counts: dict[str, int] = {}

        def record(item: FusionRecord) -> None:
            payload = {
                "pattern": item.pattern,
                "fused_node": item.fused_node,
                "original_nodes": list(item.original_nodes),
                "inputs": list(item.inputs),
                "outputs": list(item.outputs),
            }
            fusion_log.append(payload)
            pattern_counts[item.pattern] = (
                pattern_counts.get(item.pattern, 0) + 1
            )

        # Run repeatedly because each successful replacement changes graph
        # topology and can expose another independent match.
        patterns = [
            (
                {"Conv", "Conv2d"},
                {"BatchNormalization", "BatchNorm"},
                "Conv+BatchNorm",
                "FusedConvBatchNorm",
            ),
            (
                {"Conv", "Conv2d"},
                {"Relu"},
                "Conv+Relu",
                "FusedConvRelu",
            ),
            (
                {"Gemm", "Linear", "MatMul"},
                {"Add"},
                "MatMul+Add",
                "FusedMatMulAdd",
            ),
            (
                {"Add"},
                {"Relu"},
                "Add+Relu",
                "FusedAddRelu",
            ),
        ]

        for first_ops, second_ops, pattern, fused_op_type in patterns:
            while True:
                match = self._find_pair(
                    graph,
                    first_ops,
                    second_ops,
                )
                if match is None:
                    break
                record(
                    self._fuse_nodes(
                        graph,
                        list(match),
                        pattern=pattern,
                        fused_op_type=fused_op_type,
                    )
                )

        while True:
            chain = self._find_layernorm_chain(graph)
            if chain is None:
                break
            record(
                self._fuse_nodes(
                    graph,
                    chain,
                    pattern="LayerNorm",
                    fused_op_type="FusedLayerNorm",
                )
            )

        graph.validate()

        return {
            "changed": bool(fusion_log),
            "stats": {
                "num_fusions": len(fusion_log),
                "pattern_counts": pattern_counts,
                "fusion_log": fusion_log,
            },
        }
