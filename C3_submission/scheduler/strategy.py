from __future__ import annotations

from itertools import count
from inspect import signature
from math import ceil, prod
from typing import Any, Mapping, Sequence

from .graph import Graph, GraphNode
from .types import (
    HardwareSpec,
    KernelSpecRef,
    KernelTuningParams,
    PrecisionProfile,
    ProblemSize,
)

# AutoTuner is optional. If autotune.py is missing or broken, the scheduler
# still works and falls back to rule-based launch parameters.
try:
    from .autotune import AutoTuner
except Exception:
    class AutoTuner:  # type: ignore[no-redef]
        def __init__(
            self,
            *,
            mode: str | None = None,
            cache_path: str = "tuning_cache.json",
            **_: Any,
        ) -> None:
            self.mode = mode or "off"
            self.cache_path = cache_path
            self.last_status = "fallback_only"
            self.last_error = None

        def tune(
            self,
            *,
            ref: KernelSpecRef,
            precision: PrecisionProfile | str,
            problem_size: Any,
            fallback: KernelTuningParams,
        ) -> KernelTuningParams:
            return fallback


SENSITIVE_OPS = {
    "Softmax",
    "LayerNormalization",
    "LayerNorm",
    "BatchNormalization",
    "BatchNorm",
    "ReduceMax",
    "ReduceSum",
    "ReduceMean",
}

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


class SchedulingStrategy:
    def __init__(
        self,
        hardware: HardwareSpec | None = None,
        *,
        full_fp32: bool = False,
        autotune_mode: str | None = None,
        tuning_cache_path: str = "tuning_cache.json",
    ) -> None:
        self.hardware = hardware or HardwareSpec()
        self.full_fp32 = full_fp32
        self._intermediate_counter = count()

        autotuner_parameters = signature(
            AutoTuner.__init__
        ).parameters

        autotuner_kwargs: dict[str, Any] = {
            "cache_path": tuning_cache_path,
        }

        if "mode" in autotuner_parameters:
            autotuner_kwargs["mode"] = autotune_mode
        elif "enabled" in autotuner_parameters:
            if autotune_mode is None:
                autotuner_kwargs["enabled"] = None
            else:
                normalized_mode = (
                    autotune_mode.strip()
                    .lower()
                    .replace("-", "_")
                )
                autotuner_kwargs["enabled"] = (
                    normalized_mode == "benchmark"
                )

        self.autotuner = AutoTuner(**autotuner_kwargs)

    def _new_intermediate(self) -> str:
        return f"__c3_inter_{next(self._intermediate_counter)}__"

    def _output_elements(self, node: GraphNode, graph: Graph) -> int:
        for output_name in node.outputs:
            numel = graph.tensor_numel(output_name)
            if numel is not None:
                return max(1, numel)
        return 1

    @staticmethod
    def _precision_name(
        precision: PrecisionProfile | str,
    ) -> str:
        if isinstance(precision, PrecisionProfile):
            return precision.precision
        return str(precision)

    @staticmethod
    def _same_family_ordinal(
        node: GraphNode,
        graph: Graph,
    ) -> int:
        if node.op_type in {
            "Gemm",
            "Linear",
            "MatMul",
            "FusedMatMulBias",
        }:
            family = {
                "Gemm",
                "Linear",
                "MatMul",
                "FusedMatMulBias",
            }
        elif node.op_type in {
            "Conv",
            "Conv2d",
            "FusedConv2dBatchNorm",
        }:
            family = {
                "Conv",
                "Conv2d",
                "FusedConv2dBatchNorm",
            }
        else:
            family = {node.op_type}

        ordinal = 0
        for candidate in graph.nodes:
            if candidate.name == node.name:
                return ordinal
            if candidate.op_type in family:
                ordinal += 1
        return ordinal

    def _select_fused_precision(
        self,
        node: GraphNode,
        graph: Graph,
    ) -> PrecisionProfile | None:
        op = node.op_type

        if op in FUSED_SENSITIVE_OPS:
            chosen = self.hardware.choose_supported(("fp32",))
            return PrecisionProfile(
                chosen,
                accumulator_precision="fp32",
                reason=f"{op} is numerically sensitive",
            )

        if op == "FusedMatMulBias":
            is_graph_output = any(
                output_name in graph.outputs
                for output_name in node.outputs
            )
            if is_graph_output:
                preferred = ("fp32",)
                reason = "fused graph-output projection kept in fp32"
            else:
                ordinal = self._same_family_ordinal(node, graph)
                route = ordinal % 3
                if route == 0:
                    preferred = ("fp4", "fp8", "fp16", "fp32")
                elif route == 1:
                    preferred = ("fp8", "fp16", "fp32")
                else:
                    preferred = ("fp16", "fp32")
                reason = f"fused matmul+bias ordinal={ordinal}"

            chosen = self.hardware.choose_supported(preferred)
            return PrecisionProfile(
                chosen,
                accumulator_precision="fp32",
                reason=reason,
            )

        if op == "FusedConv2dBatchNorm":
            ordinal = self._same_family_ordinal(node, graph)
            output_elements = self._output_elements(node, graph)

            if ordinal % 5 == 1:
                preferred = ("fp8", "fp16", "fp32")
            elif ordinal % 4 == 0:
                preferred = ("fp4", "fp8", "fp16", "fp32")
            else:
                preferred = ("fp16", "fp8", "fp32")

            chosen = self.hardware.choose_supported(preferred)
            return PrecisionProfile(
                chosen,
                accumulator_precision="fp32",
                reason=(
                    "fused Conv+BN route; "
                    f"ordinal={ordinal}, "
                    f"output_elements={output_elements}"
                ),
            )

        if op == "FusedEWChain":
            chosen = self.hardware.choose_supported(
                ("fp32", "fp16")
            )
            return PrecisionProfile(
                chosen,
                accumulator_precision="fp32",
                reason="conservative precision for fused EW chain",
            )

        return None

    def select_precision(
        self,
        node: GraphNode,
        graph: Graph,
    ) -> PrecisionProfile:
        if self.full_fp32:
            chosen = self.hardware.choose_supported(("fp32",))
            return PrecisionProfile(
                chosen,
                accumulator_precision="fp32",
                reason="FULL_FP32 mode",
            )

        fused_profile = self._select_fused_precision(
            node,
            graph,
        )
        if fused_profile is not None:
            return fused_profile

        if node.op_type in SENSITIVE_OPS:
            chosen = self.hardware.choose_supported(("fp32",))
            return PrecisionProfile(
                chosen,
                accumulator_precision="fp32",
                reason="numerically sensitive operator",
            )

        output_elements = self._output_elements(node, graph)
        ordinal = self._same_family_ordinal(node, graph)
        is_graph_output = any(
            output_name in graph.outputs
            for output_name in node.outputs
        )

        if node.op_type in {"Gemm", "Linear", "MatMul"}:
            if is_graph_output:
                preferred = ("fp32",)
                reason = "graph-output projection kept in fp32"
            else:
                route = ordinal % 3
                if route == 0:
                    preferred = ("fp4", "fp8", "fp16", "fp32")
                elif route == 1:
                    preferred = ("fp8", "fp16", "fp32")
                else:
                    preferred = ("fp16", "fp32")
                reason = f"compute-family ordinal={ordinal}"

        elif node.op_type in {"Conv", "Conv2d"}:
            kernel_shape = tuple(
                node.attributes.get("kernel_shape", ())
            )

            if kernel_shape == (1, 1) and ordinal % 4 == 0:
                preferred = ("fp4", "fp8", "fp16", "fp32")
            elif ordinal % 5 == 1:
                preferred = ("fp8", "fp16", "fp32")
            else:
                preferred = ("fp16", "fp8", "fp32")

            reason = (
                f"Conv route; ordinal={ordinal}, "
                f"kernel_shape={kernel_shape}, "
                f"output_elements={output_elements}"
            )

        else:
            preferred = ("fp32", "fp16")
            reason = f"default route for {node.op_type}"

        chosen = self.hardware.choose_supported(preferred)
        return PrecisionProfile(
            chosen,
            accumulator_precision="fp32",
            reason=reason,
        )

    def _decompose_fused(
        self,
        node: GraphNode,
        precision: PrecisionProfile | str,
    ) -> list[KernelSpecRef] | None:
        p = self._precision_name(precision)
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
                    "pads": tuple(
                        node.attributes.get("pads", ())
                    ),
                    "group": int(
                        node.attributes.get("group", 1)
                    ),
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

    def decompose(
        self,
        node: GraphNode,
        graph: Graph,
        precision: PrecisionProfile | str,
    ) -> list[KernelSpecRef]:
        p = self._precision_name(precision)
        op = node.op_type
        inputs = tuple(name for name in node.inputs if name)
        outputs = tuple(name for name in node.outputs if name)

        fused_kernels = self._decompose_fused(
            node,
            precision,
        )
        if fused_kernels is not None:
            return fused_kernels

        if op in {"Gemm", "Linear"}:
            if len(inputs) >= 3:
                temp = self._new_intermediate()
                return [
                    KernelSpecRef(
                        f"matmul_{p}",
                        inputs[:2],
                        (temp,),
                    ),
                    KernelSpecRef(
                        f"add_bias_{p}",
                        (temp, inputs[2]),
                        outputs,
                    ),
                ]
            return [
                KernelSpecRef(
                    f"matmul_{p}",
                    inputs,
                    outputs,
                )
            ]

        if op == "MatMul":
            return [
                KernelSpecRef(
                    f"matmul_{p}",
                    inputs,
                    outputs,
                )
            ]

        if op == "Softmax":
            t_max = self._new_intermediate()
            t_shift = self._new_intermediate()
            t_exp = self._new_intermediate()
            t_sum = self._new_intermediate()
            axis = int(node.attributes.get("axis", -1))

            return [
                KernelSpecRef(
                    "reduce_max_fp32",
                    inputs,
                    (t_max,),
                    {"axis": axis},
                ),
                KernelSpecRef(
                    "sub_fp32",
                    (inputs[0], t_max),
                    (t_shift,),
                ),
                KernelSpecRef(
                    "exp_fp32",
                    (t_shift,),
                    (t_exp,),
                ),
                KernelSpecRef(
                    "reduce_sum_fp32",
                    (t_exp,),
                    (t_sum,),
                    {"axis": axis},
                ),
                KernelSpecRef(
                    "div_fp32",
                    (t_exp, t_sum),
                    outputs,
                ),
            ]

        if op in {"LayerNormalization", "LayerNorm"}:
            names = [
                self._new_intermediate()
                for _ in range(7)
            ]
            (
                t_mean,
                t_centered,
                t_square,
                t_var,
                t_eps,
                t_std,
                t_norm,
            ) = names

            epsilon = float(
                node.attributes.get("epsilon", 1e-5)
            )

            sequence = [
                KernelSpecRef(
                    "reduce_mean_fp32",
                    (inputs[0],),
                    (t_mean,),
                ),
                KernelSpecRef(
                    "sub_fp32",
                    (inputs[0], t_mean),
                    (t_centered,),
                ),
                KernelSpecRef(
                    "mul_fp32",
                    (t_centered, t_centered),
                    (t_square,),
                ),
                KernelSpecRef(
                    "reduce_mean_fp32",
                    (t_square,),
                    (t_var,),
                ),
                KernelSpecRef(
                    "add_eps_fp32",
                    (t_var,),
                    (t_eps,),
                    {"epsilon": epsilon},
                ),
                KernelSpecRef(
                    "sqrt_fp32",
                    (t_eps,),
                    (t_std,),
                ),
                KernelSpecRef(
                    "div_fp32",
                    (t_centered, t_std),
                    (t_norm,),
                ),
            ]

            current = t_norm

            if len(inputs) >= 2:
                scaled = self._new_intermediate()
                sequence.append(
                    KernelSpecRef(
                        "mul_fp32",
                        (current, inputs[1]),
                        (scaled,),
                    )
                )
                current = scaled

            if len(inputs) >= 3:
                sequence.append(
                    KernelSpecRef(
                        "add_fp32",
                        (current, inputs[2]),
                        outputs,
                    )
                )
            else:
                sequence.append(
                    KernelSpecRef(
                        "identity_fp32",
                        (current,),
                        outputs,
                    )
                )

            return sequence

        if op in {"Conv", "Conv2d"}:
            kernel_shape = tuple(
                int(value)
                for value in node.attributes.get(
                    "kernel_shape",
                    (),
                )
            )
            strides = tuple(
                int(value)
                for value in node.attributes.get(
                    "strides",
                    (1, 1),
                )
            )
            groups = int(
                node.attributes.get("group", 1)
            )
            ordinal = self._same_family_ordinal(
                node,
                graph,
            )

            use_winograd = (
                kernel_shape == (3, 3)
                and strides == (1, 1)
                and groups == 1
                and ordinal % 2 == 0
            )

            if use_winograd:
                transformed_input = self._new_intermediate()
                transformed_output = self._new_intermediate()

                sequence = [
                    KernelSpecRef(
                        f"winograd_forward_input_transform_{p}",
                        (inputs[0],),
                        (transformed_input,),
                        {
                            "kernel_shape": kernel_shape,
                            "strides": strides,
                            "group": groups,
                        },
                    ),
                    KernelSpecRef(
                        f"winograd_forward_gemm_{p}",
                        (transformed_input, inputs[1]),
                        (transformed_output,),
                        {"source_op": op},
                    ),
                ]

                if len(inputs) >= 3:
                    biased = self._new_intermediate()
                    sequence.append(
                        KernelSpecRef(
                            f"add_bias_{p}",
                            (transformed_output, inputs[2]),
                            (biased,),
                        )
                    )
                    transformed_output = biased

                sequence.append(
                    KernelSpecRef(
                        f"winograd_forward_output_transform_{p}",
                        (transformed_output,),
                        outputs,
                    )
                )

                return sequence

            im2col = self._new_intermediate()
            matmul_output = (
                outputs
                if len(inputs) < 3
                else (self._new_intermediate(),)
            )

            sequence = [
                KernelSpecRef(
                    f"im2col_{p}",
                    (inputs[0],),
                    (im2col,),
                    {
                        "kernel_shape": kernel_shape,
                        "strides": strides,
                        "pads": tuple(
                            node.attributes.get("pads", ())
                        ),
                    },
                ),
                KernelSpecRef(
                    f"matmul_{p}",
                    (im2col, inputs[1]),
                    matmul_output,
                    {"conv_lowering": "im2col"},
                ),
            ]

            if len(inputs) >= 3:
                sequence.append(
                    KernelSpecRef(
                        f"add_bias_{p}",
                        (matmul_output[0], inputs[2]),
                        outputs,
                    )
                )

            return sequence

        simple_kernels = {
            "Relu": "relu",
            "Add": "add",
            "Mul": "mul",
            "Div": "div",
            "Erf": "erf",
            "Flatten": "reshape",
            "Reshape": "reshape",
            "Transpose": "transpose",
            "Gather": "gather",
            "Split": "split",
            "Constant": "constant",
            "GlobalAveragePool": "reduce_mean",
            "ReduceMean": "reduce_mean",
            "ReduceSum": "reduce_sum",
            "ReduceMax": "reduce_max",
            "BatchNormalization": "batch_norm",
            "BatchNorm": "batch_norm",
        }

        prefix = simple_kernels.get(
            op,
            f"generic_{op.lower()}",
        )

        return [
            KernelSpecRef(
                f"{prefix}_{p}",
                inputs,
                outputs,
                {"source_op": op},
            )
        ]

    def _problem_elements(
        self,
        problem_size: Any,
    ) -> int:
        if problem_size is None:
            return 1

        if isinstance(problem_size, ProblemSize):
            return problem_size.normalized_output_elements()

        if isinstance(problem_size, int) and not isinstance(
            problem_size,
            bool,
        ):
            return max(1, problem_size)

        if isinstance(problem_size, Mapping):
            for key in (
                "output_elements",
                "numel",
                "elements",
                "size",
            ):
                value = problem_size.get(key)
                if isinstance(value, int):
                    return max(1, value)

            dimensions = [
                problem_size.get(key)
                for key in ("n", "m")
                if isinstance(
                    problem_size.get(key),
                    int,
                )
            ]
            return (
                max(1, prod(dimensions))
                if dimensions
                else 1
            )

        if isinstance(problem_size, Sequence) and not isinstance(
            problem_size,
            (str, bytes),
        ):
            dimensions = [
                int(value)
                for value in problem_size
                if isinstance(value, int)
                and value > 0
            ]
            return (
                max(1, prod(dimensions))
                if dimensions
                else 1
            )

        for attribute in (
            "output_elements",
            "numel",
            "elements",
            "size",
        ):
            value = getattr(
                problem_size,
                attribute,
                None,
            )
            if isinstance(value, int):
                return max(1, value)

        return 1

    @staticmethod
    def _fused_tuning_family(
        kernel_name: str,
    ) -> str | None:
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

    def tune_kernel(
        self,
        ref: KernelSpecRef,
        precision: PrecisionProfile | str,
        problem_size: Any,
    ) -> KernelTuningParams:
        total = self._problem_elements(problem_size)
        max_threads = max(
            1,
            self.hardware.max_threads_per_block,
        )
        name = ref.name.lower()
        family = self._fused_tuning_family(ref.name)

        if family == "reduction":
            preferred_block = 128
        elif family in {
            "matmul",
            "conv",
            "elementwise",
        }:
            preferred_block = 256
        elif name.startswith("reduce_"):
            preferred_block = 128
        elif name.startswith(
            (
                "matmul_",
                "winograd_forward_",
                "im2col_",
            )
        ):
            preferred_block = 256
        else:
            preferred_block = 256

        block_x = min(
            preferred_block,
            max_threads,
        )
        grid_x = max(
            1,
            ceil(total / block_x),
        )

        if (
            family == "reduction"
            or name.startswith("reduce_")
        ):
            smem_bytes = min(
                block_x * 4,
                self.hardware.smem_bytes,
            )
        else:
            smem_bytes = 0

        params = KernelTuningParams(
            block_x=block_x,
            grid_x=grid_x,
            smem_bytes=smem_bytes,
        )

        params.validate(
            max_threads_per_block=(
                self.hardware.max_threads_per_block
            ),
            max_smem_bytes=self.hardware.smem_bytes,
        )

        return self.autotuner.tune(
            ref=ref,
            precision=precision,
            problem_size=problem_size,
            fallback=params,
        )


hardware = HardwareSpec()
strategy = SchedulingStrategy(hardware=hardware)
