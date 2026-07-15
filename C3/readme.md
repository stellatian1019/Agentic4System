# Agentic4System Track-C C3 Submission

本目录为 C3 算子调度与模型部署赛题的完整提交代码，包含：

- C3.1：ONNX 计算图解析与 DAG JSON 导出
- C3.2：精度选择、算子分解与 Kernel 参数选择
- C3.3：计算图融合与优化
- C3.4：内存规划、权重延迟加载、异步预取与 Stream 调度
- C3.5：基于 CuPy 的 GPU 模型推理 Worker

## 1. 目录结构

```text
C3/
├── readme.md
├── export_dag.py
├── infer_worker.py
├── scheduler/
│   ├── graph.py
│   ├── onnx_importer.py
│   ├── strategy.py
│   ├── graph_passes/
│   └── memory/
└── runtime/
    ├── cupy_graph_runner.py
    ├── io.py
    └── onnx_runner.py
```

模型文件和测试数据由评测机提供，不包含在本提交目录中。

## 2. 运行环境

代码面向官方初始环境开发和测试：

```text
OS             Linux x86_64
Python         3.12.3 / 3.13.5
ONNX           1.22.0
ONNX Runtime   1.27.0
NumPy          2.5.1
CuPy           14.1.1
GPU            NVIDIA H200 MIG
```

运行依赖：

```text
numpy
onnx
onnxruntime
cupy
```

以上依赖均由官方初始环境提供，不需要联网安装其他软件包。

代码不依赖固定的 Python `site-packages` 路径，会根据当前解释器自动定位
pip 安装的 NVIDIA CUDA 动态库，以兼容 Python 3.12.3 与 3.13.5 环境。

## 3. C3.1：ONNX DAG 导出

### 命令模板

```bash
python3 export_dag.py --onnx {onnx} --output {output}
```

其中：

- `{onnx}`：评测机提供的 ONNX 模型路径
- `{output}`：需要生成的 DAG JSON 文件路径

等价调用格式：

```bash
python3 export_dag.py \
  --onnx /path/to/model.onnx \
  --output /path/to/dag.json
```

程序成功时以退出码 `0` 结束。

输出 JSON 包含：

```text
format_version
graph_inputs
graph_outputs
nodes
edges
```

`graph_inputs` 不包含 initializer 权重。节点名称、张量名称、算子类型、输入输出关系均从 ONNX 模型中提取。

程序支持普通 ONNX 模型及使用 `.onnx.data` 的 external-data 模型。

## 4. C3.2：算子调度

调度器实现在 `scheduler/strategy.py` 中，提供：

```python
select_precision(node, graph)
decompose(node, graph, precision)
tune_kernel(kernel, precision, problem_size)
```

实现内容包括：

- FP32、FP16、FP8、FP4 精度选择
- 数值敏感算子精度保护
- 复杂算子 Kernel 分解
- 中间张量显式追踪
- `block_x`、`grid_x`、`smem_bytes` 参数生成
- 硬件约束检查

## 5. C3.3：计算图融合

图优化代码位于：

```text
scheduler/graph_passes/
```

支持的主要融合模式包括：

- MatMul + Bias
- Conv + BatchNorm
- Elementwise Chain
- Softmax + Dropout
- Residual Add + LayerNormalization

融合会实际修改计算图，减少 Kernel launch 数量和中间张量数量。

## 6. C3.4：内存规划与调度

内存和执行计划代码位于：

```text
scheduler/memory/
```

实现内容包括：

- 张量生命周期分析
- 中间张量内存复用规划
- Device Memory Pool
- 权重存储管理
- ONNX external-data 权重延迟加载
- 下一阶段权重异步预取
- Compute Stream 与 Transfer Stream 调度
- 张量最后一次使用后的及时释放

对于 BigFormer，不会在启动时将全部权重加载到 GPU。运行时根据执行步骤按需读取 `.onnx.data` 中的权重并上传至 GPU，从而避免模型初始化阶段显存不足。

默认采用单批自适应执行与全流式权重：执行计划根据激活峰值把内部批量上限设为
512，外部权重使用非缓存 CUDA 分配，消费后立即归还。公开 BigFormer 上该策略将
峰值显存从接近 16 GiB 降至约 3.3 GiB。可通过
`C3_STREAM_BATCH_SIZE`、`C3_WEIGHT_CACHE_GIB`、`C3_GPU_RESERVE_GIB` 和
`C3_ACTIVATION_POOL_GIB` 覆盖默认策略。外部权重在第一次 warmup 时懒加载到
pinned host memory，以加速后续 H2D；主存受限时可设置 `C3_PIN_HOST_WEIGHTS=0`。
维度达到 16384 的 FFN 扩张 MatMul 默认使用 TF32 Tensor Core，FFN 收缩、注意力
投影、注意力内部 MatMul 和最终 logits 投影保持 FP32；可设置
`C3_ENABLE_TF32=0` 强制全 FP32。

运行时会识别 ONNX 导出的精确 GELU 链，将 Bias Add、Div、Erf 和后续逐元素运算
合并为单个 FP32 kernel。只有输入张量不存在其他消费者时才原地复用激活内存；
否则自动保留原图执行，从而避免共享张量被覆盖。默认激活池上限为 `2.75 GiB`。

普通 CUDA EP 模型默认禁用 TF32，并使用 `kSameAsRequested` arena 扩展策略，在不
改变数值结果和吞吐的情况下减少 ResNet 峰值显存。可用
`C3_CUDA_ARENA_STRATEGY` 覆盖内存策略；`C3_ORT_USE_TF32=1` 仅作为显式实验开关，
因为公开 ResNet 在 TF32 下不能通过 `1e-3` 精度门禁。

## 7. C3.5：GPU 推理 Worker

### 启动命令

```bash
python3 infer_worker.py
```

评测命令模板：

```text
python3 infer_worker.py
```

Worker 遵循官方 `C35_WORKER_PROTOCOL.md`：

1. 进程启动并完成初始化；
2. 在标准输出打印：

```text
READY
```

3. 持续从标准输入读取评测请求；
4. 完成推理并输出协议规定的 JSON 响应；
5. 日志和诊断信息写入标准错误，不污染标准输出；
6. 支持在同一 Worker 进程中连续处理多个请求。

## 8. 支持的模型

支持公开评测中的：

```text
mlp_v1
resnet_v1
transformer_v1
bigformer_v1
```

支持动态 batch size。每次推理结果的第 0 维顺序与输入样本顺序一致。

## 9. 支持的算子

当前支持18种正式评测算子：

```text
Add
Constant
Conv
Div
Erf
Flatten
Gather
Gemm
GlobalAveragePool
Identity
LayerNormalization
MatMul
Mul
Relu
Reshape
Softmax
Split
Transpose
```

GPU 数值计算统一使用 CuPy。

## 10. 数值与输出

推理输出使用 `float32`，并按照官方协议写入指定输出目录。

模型输出名称保持为 ONNX 模型定义的名称，例如：

```text
logits
```

## 11. 公开环境验证摘要

使用官方公开模型、数据和持久化 Worker 协议验证：

- 调度、融合和内存规划单元测试：43/43 通过；
- MLP、ResNet、Transformer、BigFormer 均通过 `rtol=atol=1e-3`；
- BigFormer（请求 batch 64、内部自适应 batch 512）在 2 次 warmup + 5 次计时下，
  中位数约 `33.740s`，稳定轮范围 `33.736–33.774s`，峰值显存约 `3.304 GiB`；
- BigFormer 的公开 golden 最大绝对误差约 `5.49e-4`；设置
  `C3_ENABLE_TF32=0` 可切换到严格 FP32 路径。
- ResNet 在相同正式口径下中位数约 `4.429s`，峰值显存约 `2.696 GiB`，最大绝对
  误差约 `9.54e-6`。
```

实现以官方 FP32 参考输出和以下误差要求为正确性目标：

```text
rtol = 1e-3
atol = 1e-3
```

## 11. 使用说明

评测时应首先进入本目录：

```bash
cd C3
```

C3.1：

```bash
python3 export_dag.py --onnx {onnx} --output {output}
```

C3.5：

```bash
python3 infer_worker.py
```

程序不需要访问互联网，不会下载模型、权重或额外依赖。
