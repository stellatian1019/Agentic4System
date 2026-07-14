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
    └── worker_protocol.py
```

模型文件和测试数据由评测机提供，不包含在本提交目录中。

## 2. 运行环境

代码面向官方初始环境开发和测试：

```text
OS             Linux x86_64
Python         3.12.3
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