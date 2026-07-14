# Agentic4System C3 Submission

This directory is the complete C3 submission. It targets the official Python 3.12 / CuPy 14 / ONNX 1.22 / ONNX Runtime 1.27 environment and requires no network access.

## Command templates

C3.1:

```text
python3 export_dag.py --onnx {onnx} --output {output}
```

C3.5 persistent worker:

```text
python3 infer_worker.py
```

The worker writes only `READY` and one JSON result per task to stdout. Diagnostics go to stderr. Send `{"cmd":"exit"}` to stop it.

## Implementation overview

- **C3.1**: ONNX-to-DAG JSON exporter, including external-data models.
- **C3.2**: public `import_onnx_graph`, precision selection, decomposition, intermediate tensors and complete tuning parameters.
- **C3.3**: graph-pass pipeline and five required fusion families with numerical regression tests.
- **C3.4**: best-fit/coalescing pools, tensor lifetime reuse, execution plans, dependency-aware multi-stream schedules, and asynchronous weight prefetch.
- **C3.5**: ORT CUDA for ordinary models and a CuPy graph runner for external-data BigFormer.

BigFormer does not bulk-upload its approximately 18 GB of weights. The runtime builds the C3.4 execution/memory plan, mmaps external tensors, reserves plan-derived activation headroom, caches only weights that fit, and overlaps next-weight H2D transfer with current-node computation. `Identity` is implemented as a zero-copy alias/pass-through, so all 18 operators in the current specification are covered.

## Public self-check

Run unit/regression tests:

```bash
python3 -m unittest discover -s tests
python3 final_submission_audit.py
```

Run the official persistent-worker checker from this directory:

```bash
python3 /workspace/C3/testcases/selfcheck_worker.py \  --worker "python3 infer_worker.py" \  --models mlp_v1 resnet_v1 transformer_v1 bigformer_v1 \  --batch-size 64 --check-precision
```

For a quick BigFormer check:

```bash
python3 /workspace/C3/testcases/selfcheck_worker.py \  --worker "python3 infer_worker.py" \  --models bigformer_v1 --warmup 2 --timed 1 \  --batch-size 64 --check-precision
```

The current official spec uses two warmups and five timed tasks, and uses the median of the five timed tasks. Performance ranking uses ResNet (20%) and BigFormer (80%); all four models are checked for numerical correctness.

## Optional runtime controls

- `C3_GPU_RESERVE_GIB`: override GPU headroom. Without it, the runtime uses the C3.4 activation peak plus a conservative 2 GiB safety margin.
- `C3_WEIGHT_CACHE_GIB`: cap the persistent BigFormer weight cache.

Do not package model files, generated outputs, caches, or virtual environments with the source submission.


## C3.2 / C3.3 local conformance check

scheduler/benchmark.py is an internal CuPy autotuning helper. The official
submission does not invoke it. Run this report before packaging:

python3 validate_c32_c33.py /workspace/C3/testcases/models/mlp_v1.onnx /workspace/C3/testcases/models/resnet_v1.onnx /workspace/C3/testcases/models/transformer_v1.onnx --output c32_c33_conformance.json

The BigFormer runtime consumes the C3.4 stream schedule and transfer events.
One compute stream is the default because it was faster on public BigFormer.
Dependency-safe multi-compute-stream execution remains available with:

C3_NUM_COMPUTE_STREAMS=2 python3 infer_worker.py

No package outside the official Python 3.12 environment is required. Runtime
third-party imports are limited to NumPy, ONNX, ONNX Runtime and CuPy.
