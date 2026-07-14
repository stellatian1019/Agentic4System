from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from ..graph import Graph, GraphNode
from ..types import TensorInfo


ELEMENTWISE_OPS = {
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Relu",
    "LeakyRelu",
    "Sigmoid",
    "Tanh",
    "Erf",
    "Exp",
    "Sqrt",
    "Clip",
}


@dataclass(frozen=True)
class FusionMatch:
    pattern: str
    nodes: tuple[GraphNode, ...]


class FusionPass:
    """
    C3.3 fusion pass at the required path:
        scheduler/graph_passes/fusion.py

    Exact scored pattern names:
    - FusedMatMulBias
    - FusedConv2dBatchNorm
    - FusedEWChain
    - FusedSoftmaxDropout
    - FusedResidualNorm
    """

    name = "Fusion"

    def __init__(self) -> None:
        self._counter = 0

    def _new_name(self, pattern: str) -> str:
        name = f"__c3_fused_{pattern}_{self._counter}"
        self._counter += 1
        return name

    @staticmethod
    def _single_consumer(graph: Graph, tensor_name: str) -> GraphNode | None:
        consumers = graph.get_consumers(tensor_name)
        return consumers[0] if len(consumers) == 1 else None

    @staticmethod
    def _external_inputs(nodes: Sequence[GraphNode]) -> list[str]:
        produced = {
            tensor_name
            for node in nodes
            for tensor_name in node.outputs
            if tensor_name
        }
        result: list[str] = []
        seen: set[str] = set()
        for node in nodes:
            for tensor_name in node.inputs:
                if (
                    tensor_name
                    and tensor_name not in produced
                    and tensor_name not in seen
                ):
                    result.append(tensor_name)
                    seen.add(tensor_name)
        return result

    @staticmethod
    def _external_outputs(
        graph: Graph,
        nodes: Sequence[GraphNode],
    ) -> list[str]:
        node_names = {node.name for node in nodes}
        result: list[str] = []
        seen: set[str] = set()

        for node in nodes:
            for tensor_name in node.outputs:
                consumers = graph.get_consumers(tensor_name)
                escapes = (
                    tensor_name in graph.outputs
                    or any(c.name not in node_names for c in consumers)
                )
                if escapes and tensor_name not in seen:
                    result.append(tensor_name)
                    seen.add(tensor_name)

        if not result:
            result.extend(nodes[-1].outputs)
        return result

    @staticmethod
    def _replace_contiguous(
        graph: Graph,
        nodes: Sequence[GraphNode],
        fused: GraphNode,
    ) -> None:
        selected = {node.name for node in nodes}
        positions = [
            index
            for index, node in enumerate(graph.nodes)
            if node.name in selected
        ]
        if len(positions) != len(nodes):
            raise ValueError("Fusion nodes are not all present in graph")

        insertion = min(positions)
        new_nodes = [
            node for node in graph.nodes if node.name not in selected
        ]
        new_nodes.insert(insertion, fused)
        graph.nodes = new_nodes
        graph.rebuild_index()

    def _generic_fuse(
        self,
        graph: Graph,
        match: FusionMatch,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        nodes = list(match.nodes)
        inputs = self._external_inputs(nodes)
        outputs = self._external_outputs(graph, nodes)
        fused_name = self._new_name(match.pattern)

        fused_attributes: dict[str, Any] = {
            "fusion_pattern": match.pattern,
            "original_nodes": tuple(node.name for node in nodes),
            "original_op_types": tuple(node.op_type for node in nodes),
        }
        if attributes:
            fused_attributes.update(attributes)

        fused = GraphNode(
            name=fused_name,
            op_type=match.pattern,
            inputs=inputs,
            outputs=outputs,
            attributes=fused_attributes,
        )
        self._replace_contiguous(graph, nodes, fused)

        return {
            "pattern": match.pattern,
            "fused_node": fused_name,
            "original_nodes": [node.name for node in nodes],
            "inputs": inputs,
            "outputs": outputs,
            "launches_before": len(nodes),
            "launches_after": 1,
            "internal_buffers_removed": max(0, len(nodes) - 1),
        }

    def _find_pair(
        self,
        graph: Graph,
        first_ops: set[str],
        second_ops: set[str],
        pattern: str,
    ) -> FusionMatch | None:
        for first in graph.topological_nodes():
            if first.op_type not in first_ops or len(first.outputs) != 1:
                continue
            intermediate = first.outputs[0]
            if intermediate in graph.outputs:
                continue
            second = self._single_consumer(graph, intermediate)
            if second is not None and second.op_type in second_ops:
                return FusionMatch(pattern, (first, second))
        return None

    def _find_matmul_bias(self, graph: Graph) -> FusionMatch | None:
        for matmul in graph.topological_nodes():
            if matmul.op_type not in {"MatMul", "Gemm", "Linear"}:
                continue
            if len(matmul.outputs) != 1:
                continue
            output = matmul.outputs[0]
            if output in graph.outputs:
                continue
            add = self._single_consumer(graph, output)
            if add is None or add.op_type not in {"Add", "AddBias"}:
                continue
            # For MatMul->AddBias, the Add should consume the MatMul result plus
            # one external tensor (normally a bias initializer).
            if output not in add.inputs or len(add.inputs) < 2:
                continue
            return FusionMatch("FusedMatMulBias", (matmul, add))
        return None

    def _find_conv_bn(self, graph: Graph) -> FusionMatch | None:
        return self._find_pair(
            graph,
            {"Conv", "Conv2d"},
            {"BatchNormalization", "BatchNorm"},
            "FusedConv2dBatchNorm",
        )

    def _find_softmax_dropout(self, graph: Graph) -> FusionMatch | None:
        return self._find_pair(
            graph,
            {"Softmax"},
            {"Dropout"},
            "FusedSoftmaxDropout",
        )

    def _find_residual_norm(self, graph: Graph) -> FusionMatch | None:
        return self._find_pair(
            graph,
            {"Add"},
            {"LayerNormalization", "LayerNorm"},
            "FusedResidualNorm",
        )

    def _find_ew_chain(self, graph: Graph) -> FusionMatch | None:
        for start in graph.topological_nodes():
            if start.op_type not in ELEMENTWISE_OPS:
                continue

            chain = [start]
            current = start
            while len(chain) < 5 and len(current.outputs) == 1:
                intermediate = current.outputs[0]
                if intermediate in graph.outputs:
                    break
                nxt = self._single_consumer(graph, intermediate)
                if nxt is None or nxt.op_type not in ELEMENTWISE_OPS:
                    break
                chain.append(nxt)
                current = nxt

            if len(chain) >= 2:
                return FusionMatch("FusedEWChain", tuple(chain))
        return None

    @staticmethod
    def _initializer(graph: Graph, name: str) -> np.ndarray:
        if name not in graph.initializers:
            raise KeyError(f"Missing initializer {name!r}")
        return np.asarray(graph.initializers[name])

    def _fold_conv_bn(
        self,
        graph: Graph,
        match: FusionMatch,
    ) -> dict[str, Any]:
        """
        Numerically fold explicit BatchNorm parameters into Conv weights.

        Conv inputs:
            X, W, optional B
        BatchNorm inputs:
            conv_y, scale(gamma), B(beta), mean, var
        """
        conv, bn = match.nodes
        if len(conv.inputs) < 2:
            raise ValueError("Conv node must have X and W inputs")
        if len(bn.inputs) < 5:
            raise ValueError(
                "BatchNormalization must have scale, bias, mean and variance"
            )

        x_name = conv.inputs[0]
        w_name = conv.inputs[1]
        conv_bias_name = conv.inputs[2] if len(conv.inputs) >= 3 else None

        gamma_name, beta_name, mean_name, var_name = bn.inputs[1:5]
        weight = self._initializer(graph, w_name)
        gamma = self._initializer(graph, gamma_name)
        beta = self._initializer(graph, beta_name)
        mean = self._initializer(graph, mean_name)
        variance = self._initializer(graph, var_name)

        if weight.ndim < 2:
            raise ValueError("Conv weight must have output-channel dimension")
        out_channels = weight.shape[0]
        for name, value in (
            ("gamma", gamma),
            ("beta", beta),
            ("mean", mean),
            ("variance", variance),
        ):
            if value.size != out_channels:
                raise ValueError(
                    f"BN {name} size {value.size} != Conv output channels "
                    f"{out_channels}"
                )

        if conv_bias_name is None:
            conv_bias = np.zeros(out_channels, dtype=weight.dtype)
        else:
            conv_bias = self._initializer(graph, conv_bias_name).astype(
                weight.dtype,
                copy=False,
            )

        epsilon = float(bn.attributes.get("epsilon", 1e-5))
        factor = gamma.astype(np.float64) / np.sqrt(
            variance.astype(np.float64) + epsilon
        )

        reshape = (out_channels,) + (1,) * (weight.ndim - 1)
        merged_weight = (
            weight.astype(np.float64) * factor.reshape(reshape)
        ).astype(weight.dtype)
        merged_bias = (
            (conv_bias.astype(np.float64) - mean.astype(np.float64))
            * factor
            + beta.astype(np.float64)
        ).astype(weight.dtype)

        fused_name = self._new_name("FusedConv2dBatchNorm")
        merged_w_name = f"{fused_name}__weight"
        merged_b_name = f"{fused_name}__bias"

        graph.initializers[merged_w_name] = merged_weight
        graph.initializers[merged_b_name] = merged_bias
        graph.tensors[merged_w_name] = TensorInfo(
            merged_w_name,
            graph.tensors.get(w_name, TensorInfo(w_name)).dtype,
            tuple(int(v) for v in merged_weight.shape),
        )
        graph.tensors[merged_b_name] = TensorInfo(
            merged_b_name,
            graph.tensors.get(w_name, TensorInfo(w_name)).dtype,
            tuple(int(v) for v in merged_bias.shape),
        )

        outputs = self._external_outputs(graph, [conv, bn])
        fused_attributes = dict(conv.attributes)
        fused_attributes.update(
            {
                "fusion_pattern": "FusedConv2dBatchNorm",
                "original_nodes": (conv.name, bn.name),
                "batchnorm_epsilon": epsilon,
                "weights_folded": True,
            }
        )

        fused = GraphNode(
            name=fused_name,
            op_type="FusedConv2dBatchNorm",
            inputs=[x_name, merged_w_name, merged_b_name],
            outputs=outputs,
            attributes=fused_attributes,
        )
        self._replace_contiguous(graph, [conv, bn], fused)

        return {
            "pattern": "FusedConv2dBatchNorm",
            "fused_node": fused_name,
            "original_nodes": [conv.name, bn.name],
            "inputs": list(fused.inputs),
            "outputs": outputs,
            "launches_before": 2,
            "launches_after": 1,
            "internal_buffers_removed": 1,
            "weights_folded": True,
            "merged_weight": merged_w_name,
            "merged_bias": merged_b_name,
        }

    def _consume_all(
        self,
        graph: Graph,
        finder: Any,
        fusion_log: list[dict[str, Any]],
        *,
        conv_bn: bool = False,
    ) -> None:
        while True:
            match = finder(graph)
            if match is None:
                return
            if conv_bn:
                record = self._fold_conv_bn(graph, match)
            else:
                record = self._generic_fuse(graph, match)
            fusion_log.append(record)

    def run(self, graph: Graph) -> dict[str, Any]:
        raw_node_count = len(graph.nodes)
        fusion_log: list[dict[str, Any]] = []

        # More specific patterns must run before the generic EW chain.
        self._consume_all(graph, self._find_conv_bn, fusion_log, conv_bn=True)
        self._consume_all(graph, self._find_matmul_bias, fusion_log)
        self._consume_all(graph, self._find_softmax_dropout, fusion_log)
        self._consume_all(graph, self._find_residual_norm, fusion_log)
        self._consume_all(graph, self._find_ew_chain, fusion_log)

        graph.validate()

        pattern_counts: dict[str, int] = {}
        for record in fusion_log:
            pattern = record["pattern"]
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        launches_before = raw_node_count
        launches_after = len(graph.nodes)
        raw_buffers = max(0, raw_node_count - 1)
        optimized_buffers = max(0, launches_after - 1)

        return {
            "changed": bool(fusion_log),
            "stats": {
                "num_fusions": len(fusion_log),
                "pattern_counts": pattern_counts,
                "fusion_log": fusion_log,
                "raw_launches": launches_before,
                "optimized_launches": launches_after,
                "launch_reduction": (
                    (launches_before - launches_after) / launches_before
                    if launches_before
                    else 0.0
                ),
                "raw_buffers": raw_buffers,
                "optimized_buffers": optimized_buffers,
                "buffer_reduction": (
                    (raw_buffers - optimized_buffers) / raw_buffers
                    if raw_buffers
                    else 0.0
                ),
            },
        }
