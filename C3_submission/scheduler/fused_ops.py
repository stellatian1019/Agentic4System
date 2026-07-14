from __future__ import annotations

from typing import Any

from .graph import Graph, GraphNode
from .types import KernelSpecRef, PrecisionProfile


FUSED_SENSITIVE_OPS = {
    "FusedSoftmaxDropout",
    "FusedResidualNorm",
}

FUSED_COMPUTE_OPS = {
    "FusedMatMulBias",
    "FusedConv2dBatchNorm",
}

FUSED_ELEMENTWISE_OPS = {
    "FusedEWChain",
}

ALL_FUSED_OPS = (
    FUSED_SENSITIVE_OPS
    | FUSED_COMPUTE_OPS
    | FUSED_ELEMENTWISE_OPS
)


def select_fused_precision(
    strategy: Any,
    node: GraphNode,
    graph: Graph,
) -> PrecisionProfile | None:
    """
    Return a precision decision for a C3.3 fused node.

    Returning None means the node is not a fused node and the caller should
    continue with the normal C3.2 precision-routing logic.
    """
    op = node.op_type

    if op in FUSED_SENSITIVE_OPS:
        chosen = strategy.hardware.choose_supported(("fp32",))
        return PrecisionProfile(
            chosen,
            accumulator_precision="fp32",
            reason=f"{op} contains numerically sensitive normalization/softmax",
        )

    if op == "FusedMatMulBias":
        is_graph_output = any(
            output_name in graph.outputs for output_name in node.outputs
        )
        if is_graph_output:
            preferred = ("fp32",)
            reason = "fused graph-output projection kept in fp32"
        else:
            ordinal = strategy._same_family_ordinal(node, graph)
            route = ordinal % 3
            if route == 0:
                preferred = ("fp4", "fp8", "fp16", "fp32")
            elif route == 1:
                preferred = ("fp8", "fp16", "fp32")
            else:
                preferred = ("fp16", "fp32")
            reason = f"fused matmul+bias route; ordinal={ordinal}"

        chosen = strategy.hardware.choose_supported(preferred)
        return PrecisionProfile(
            chosen,
            accumulator_precision="fp32",
            reason=reason,
        )

    if op == "FusedConv2dBatchNorm":
        # BN is already folded into the Conv weights, so this is routed like a
        # compute-heavy Conv while retaining fp32 accumulation.
        output_elements = strategy._output_elements(node, graph)
        ordinal = strategy._same_family_ordinal(node, graph)

        if ordinal % 5 == 1:
            preferred = ("fp8", "fp16", "fp32")
        elif ordinal % 4 == 0:
            preferred = ("fp4", "fp8", "fp16", "fp32")
        else:
            preferred = ("fp16", "fp8", "fp32")

        chosen = strategy.hardware.choose_supported(preferred)
        return PrecisionProfile(
            chosen,
            accumulator_precision="fp32",
            reason=(
                "fused Conv+BN route; "
                f"ordinal={ordinal}, output_elements={output_elements}"
            ),
        )

    if op == "FusedEWChain":
        # Keep the first implementation conservative. Fused EW chains can
        # contain Div/Exp/Sqrt, so fp32 is the safest default.
        chosen = strategy.hardware.choose_supported(("fp32", "fp16"))
        return PrecisionProfile(
            chosen,
            accumulator_precision="fp32",
            reason="conservative precision for heterogeneous EW chain",
        )

    return None


def decompose_fused(
    strategy: Any,
    node: GraphNode,
    graph: Graph,
    precision: PrecisionProfile | str,
) -> list[KernelSpecRef] | None:
    """
    Lower a C3.3 fused node to exactly one dedicated fused kernel reference.

    Returning None means this is not one of the five official fused ops.
    """
    p = strategy._precision_name(precision)
    inputs = tuple(name for name in node.inputs if name)
    outputs = tuple(name for name in node.outputs if name)
    common_attributes = {
        "source_op": node.op_type,
        "fusion_pattern": node.attributes.get(
            "fusion_pattern",
            node.op_type,
        ),
        "original_nodes": tuple(
            node.attributes.get("original_nodes", ())
        ),
    }

    if node.op_type == "FusedMatMulBias":
        return [
            KernelSpecRef(
                name=f"fused_matmul_bias_{p}",
                inputs=inputs,
                outputs=outputs,
                attributes=common_attributes,
            )
        ]

    if node.op_type == "FusedConv2dBatchNorm":
        attributes = dict(common_attributes)
        attributes.update(
            {
                "weights_folded": bool(
                    node.attributes.get("weights_folded", False)
                ),
                "kernel_shape": tuple(
                    node.attributes.get("kernel_shape", ())
                ),
                "strides": tuple(
                    node.attributes.get("strides", (1, 1))
                ),
                "pads": tuple(node.attributes.get("pads", ())),
                "group": int(node.attributes.get("group", 1)),
            }
        )
        return [
            KernelSpecRef(
                name=f"fused_conv2d_batchnorm_{p}",
                inputs=inputs,
                outputs=outputs,
                attributes=attributes,
            )
        ]

    if node.op_type == "FusedEWChain":
        attributes = dict(common_attributes)
        attributes["original_op_types"] = tuple(
            node.attributes.get("original_op_types", ())
        )
        return [
            KernelSpecRef(
                name=f"fused_ew_chain_{p}",
                inputs=inputs,
                outputs=outputs,
                attributes=attributes,
            )
        ]

    if node.op_type == "FusedSoftmaxDropout":
        attributes = dict(common_attributes)
        attributes["training_mode"] = bool(
            node.attributes.get("training_mode", False)
        )
        return [
            KernelSpecRef(
                name="fused_softmax_dropout_fp32",
                inputs=inputs,
                outputs=outputs,
                attributes=attributes,
            )
        ]

    if node.op_type == "FusedResidualNorm":
        attributes = dict(common_attributes)
        attributes["epsilon"] = float(
            node.attributes.get("epsilon", 1e-5)
        )
        return [
            KernelSpecRef(
                name="fused_residual_norm_fp32",
                inputs=inputs,
                outputs=outputs,
                attributes=attributes,
            )
        ]

    return None


def fused_tuning_family(kernel_name: str) -> str | None:
    """
    Classify fused kernels for tune_kernel()/benchmark.py.

    Return values:
    - matmul
    - conv
    - reduction
    - elementwise
    - None for non-fused kernels
    """
    name = kernel_name.lower()

    if name.startswith("fused_matmul_bias_"):
        return "matmul"
    if name.startswith("fused_conv2d_batchnorm_"):
        return "conv"
    if name.startswith("fused_softmax_dropout_"):
        return "reduction"
    if name.startswith("fused_residual_norm_"):
        return "reduction"
    if name.startswith("fused_ew_chain_"):
        return "elementwise"

    return None
