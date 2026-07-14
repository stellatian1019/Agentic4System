from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scheduler import GraphPassPipeline, SchedulingStrategy, import_onnx_graph


def audit_model(model_path: Path) -> dict[str, Any]:
    graph = import_onnx_graph(model_path)
    strategy = SchedulingStrategy(autotune_mode="off")
    precision_counts: Counter[str] = Counter()
    op_counts: Counter[str] = Counter()
    kernel_counts: Counter[str] = Counter()
    intermediate_tensors: set[str] = set()

    for node in graph.nodes:
        op_counts[node.op_type] += 1
        precision = strategy.select_precision(node, graph)
        precision_counts[precision.precision] += 1
        kernels = strategy.decompose(node, graph, precision)
        if not kernels:
            raise RuntimeError(f"{node.name}: decomposition returned no kernels")

        problem_size = 1
        for output_name in node.outputs:
            numel = graph.tensor_numel(output_name)
            if isinstance(numel, int) and numel > 0:
                problem_size = numel
                break

        for kernel in kernels:
            kernel_counts[kernel.name] += 1
            intermediate_tensors.update(
                name
                for name in kernel.outputs
                if name not in node.outputs
            )
            tuning = strategy.tune_kernel(
                kernel,
                precision,
                problem_size,
            )
            tuning.validate(
                max_threads_per_block=strategy.hardware.max_threads_per_block,
                max_smem_bytes=strategy.hardware.smem_bytes,
            )

    optimized, pass_reports = GraphPassPipeline().run(graph)
    fusion_stats = pass_reports.get("Fusion", {}).get("stats", {})
    compact_fusion = {
        key: fusion_stats.get(key)
        for key in (
            "num_fusions",
            "pattern_counts",
            "raw_launches",
            "optimized_launches",
            "raw_buffers",
            "optimized_buffers",
        )
    }

    return {
        "model": str(model_path),
        "nodes": len(graph.nodes),
        "operators": dict(sorted(op_counts.items())),
        "precision_counts": dict(sorted(precision_counts.items())),
        "kernel_count": sum(kernel_counts.values()),
        "kernel_names": dict(sorted(kernel_counts.items())),
        "intermediate_tensor_count": len(intermediate_tensors),
        "optimized_nodes": len(optimized.nodes),
        "fusion": compact_fusion,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local C3.2/C3.3 conformance report (not a scoring entrypoint)."
    )
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {
        "format_version": "1.0",
        "models": [audit_model(path.resolve()) for path in args.models],
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
