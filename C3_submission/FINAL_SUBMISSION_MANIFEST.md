# C3 final submission audit

- Overall passed: **True**
- Errors: **0**
- Warnings: **0**
- Test suite passed: **True**

## Cleanup

- would delete cache: runtime/__pycache__
- would delete cache: scheduler/__pycache__
- would delete cache: tests/__pycache__
- would delete cache: scheduler/memory/__pycache__
- would delete cache: scheduler/graph_passes/__pycache__
- would delete bytecode: runtime/__pycache__/cupy_graph_runner.cpython-312.pyc
- would delete bytecode: runtime/__pycache__/onnx_runner.cpython-312.pyc
- would delete bytecode: runtime/__pycache__/__init__.cpython-312.pyc
- would delete bytecode: runtime/__pycache__/io.cpython-312.pyc
- would delete bytecode: scheduler/__pycache__/onnx_importer.cpython-312.pyc
- would delete bytecode: scheduler/__pycache__/tuning_cache.cpython-312.pyc
- would delete bytecode: scheduler/__pycache__/graph.cpython-312.pyc
- would delete bytecode: scheduler/__pycache__/benchmark.cpython-312.pyc
- would delete bytecode: scheduler/__pycache__/__init__.cpython-312.pyc
- would delete bytecode: scheduler/__pycache__/types.cpython-312.pyc
- would delete bytecode: scheduler/__pycache__/strategy.cpython-312.pyc
- would delete bytecode: scheduler/__pycache__/fused_ops.cpython-312.pyc
- would delete bytecode: scheduler/__pycache__/fusion.cpython-312.pyc
- would delete bytecode: scheduler/__pycache__/hardware.cpython-312.pyc
- would delete bytecode: scheduler/__pycache__/autotune.cpython-312.pyc
- would delete bytecode: scheduler/memory/__pycache__/planner.cpython-312.pyc
- would delete bytecode: scheduler/memory/__pycache__/plan_builder.cpython-312.pyc
- would delete bytecode: scheduler/memory/__pycache__/prefetch.cpython-312.pyc
- would delete bytecode: scheduler/memory/__pycache__/backend.cpython-312.pyc
- would delete bytecode: scheduler/memory/__pycache__/__init__.cpython-312.pyc
- would delete bytecode: scheduler/memory/__pycache__/weight_store.cpython-312.pyc
- would delete bytecode: scheduler/memory/__pycache__/pool.cpython-312.pyc
- would delete bytecode: scheduler/memory/__pycache__/execution_plan.cpython-312.pyc
- would delete bytecode: scheduler/memory/__pycache__/stream_scheduler.cpython-312.pyc
- would delete bytecode: scheduler/graph_passes/__pycache__/__init__.cpython-312.pyc
- would delete bytecode: scheduler/graph_passes/__pycache__/pipeline.cpython-312.pyc
- would delete bytecode: scheduler/graph_passes/__pycache__/fusion.cpython-312.pyc
- would delete bytecode: tests/__pycache__/test_c33_numerical.cpython-312.pyc
- would delete bytecode: tests/__pycache__/test_c34_backend_weight_store.cpython-312.pyc
- would delete bytecode: tests/__pycache__/test_c34_prefetch_stream_scheduler.cpython-312.pyc
- would delete bytecode: tests/__pycache__/test_c34_memory_foundation.cpython-312.pyc
- would delete bytecode: tests/__pycache__/test_c33_fusions.cpython-312.pyc
- would delete bytecode: tests/__pycache__/test_c34_lifetime_reuse.cpython-312.pyc
- would delete bytecode: tests/__pycache__/test_c34_execution_plan_builder.cpython-312.pyc
- would delete bytecode: tests/__pycache__/test_fused_strategy.cpython-312.pyc

## Findings

- No findings.

## Public imports

### `scheduler`

- File: `/home/mig29/agentic4system/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/C3_submission/scheduler/__init__.py`
- `Graph`: OK
- `GraphNode`: OK
- `HardwareSpec`: OK
- `KernelSpecRef`: OK
- `KernelTuningParams`: OK
- `PrecisionProfile`: OK
- `ProblemSize`: OK
- `SchedulingStrategy`: OK
- `import_onnx_graph`: OK

### `scheduler.graph_passes`

- File: `/home/mig29/agentic4system/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/C3_submission/scheduler/graph_passes/__init__.py`
- `FusionPass`: OK
- `GraphPassPipeline`: OK

## Final submission manifest

- `README.md` (3471 bytes, report-or-tool)
- `export_dag.py` (6980 bytes, report-or-tool)
- `final_submission_audit.py` (19219 bytes, report-or-tool)
- `infer_worker.py` (3570 bytes, report-or-tool)
- `runtime/__init__.py` (163 bytes, report-or-tool)
- `runtime/cupy_graph_runner.py` (19384 bytes, report-or-tool)
- `runtime/io.py` (1313 bytes, report-or-tool)
- `runtime/onnx_runner.py` (3902 bytes, report-or-tool)
- `scheduler/__init__.py` (688 bytes, production)
- `scheduler/autotune.py` (4281 bytes, production)
- `scheduler/benchmark.py` (8734 bytes, production)
- `scheduler/fused_ops.py` (7055 bytes, production)
- `scheduler/fusion.py` (8314 bytes, production)
- `scheduler/graph.py` (6300 bytes, production)
- `scheduler/graph_passes/__init__.py` (118 bytes, production)
- `scheduler/graph_passes/fusion.py` (14350 bytes, production)
- `scheduler/graph_passes/pipeline.py` (1581 bytes, production)
- `scheduler/hardware.py` (4654 bytes, production)
- `scheduler/memory/__init__.py` (1303 bytes, production)
- `scheduler/memory/backend.py` (4443 bytes, production)
- `scheduler/memory/execution_plan.py` (4474 bytes, production)
- `scheduler/memory/plan_builder.py` (10299 bytes, production)
- `scheduler/memory/planner.py` (10128 bytes, production)
- `scheduler/memory/pool.py` (9020 bytes, production)
- `scheduler/memory/prefetch.py` (7240 bytes, production)
- `scheduler/memory/stream_scheduler.py` (6014 bytes, production)
- `scheduler/memory/weight_store.py` (5033 bytes, production)
- `scheduler/onnx_importer.py` (7279 bytes, production)
- `scheduler/strategy.py` (26742 bytes, production)
- `scheduler/tuning_cache.py` (4510 bytes, production)
- `scheduler/types.py` (3337 bytes, production)
- `tests/test_c33_fusions.py` (4368 bytes, test)
- `tests/test_c33_numerical.py` (15428 bytes, test)
- `tests/test_c34_backend_weight_store.py` (4120 bytes, test)
- `tests/test_c34_execution_plan_builder.py` (5345 bytes, test)
- `tests/test_c34_lifetime_reuse.py` (5876 bytes, test)
- `tests/test_c34_memory_foundation.py` (4248 bytes, test)
- `tests/test_c34_prefetch_stream_scheduler.py` (5322 bytes, test)
- `tests/test_fused_strategy.py` (3636 bytes, test)
- `validate_c32_c33.py` (3312 bytes, report-or-tool)
