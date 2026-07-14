# Track-C C2 Runtime Submission

## 1. 项目简介

本提交实现 Track-C 的 C2 Runtime，包括完整的 AEC Runtime 动态库，以及用于性能策略选择的 DMA Agent 和 Kernel Agent。

Runtime 按照赛题规定，通过官方设备 ABI 与冻结 Kernel Image 完成计算。GEMM、AXPY、DOT、NRM2 等计算均在 AEC 设备侧执行，不在 Host/CPU 侧代算计算结果。

## 2. 提交内容

```text
C2-submission/
├── libaec.so
├── agents/
│   ├── dma_agent.py
│   └── kernel_agent.py
└── README.md
```

各文件说明如下：

| 文件 | 说明 |
|---|---|
| `libaec.so` | C2 Runtime 动态库，实现规定的 AEC Runtime C ABI |
| `agents/dma_agent.py` | DMA 策略 Agent，选择 DMA channel、chunk size、queue depth 和 zero-copy 策略 |
| `agents/kernel_agent.py` | Kernel 策略 Agent，在给定候选集合中选择满足约束的 Kernel |
| `README.md` | 提交说明、接口能力、依赖和验证方式 |

`libaec_device.so` 为官方提供的官方设备库，因此未包含在本提交中。最终运行和评分应使用官方提供的最新版本。

## 3. Runtime 功能

### 3.1 设备与错误处理

实现以下能力：

- 设备数量查询
- 设备信息与 ABI/ISA 信息查询
- Runtime 错误码与错误名称查询
- Thread-local last-error 状态
- 非法参数、非法地址和非法 Handle 检查

### 3.2 设备内存

实现以下能力：

- 设备内存分配与释放
- 分配空间复用
- Out-of-memory 检查
- 重复释放检查
- allocation-relative 地址范围验证
- 释放前等待此前关联的异步工作完成

### 3.3 数据传输

实现以下能力：

- 同步 Host-to-Device 复制
- 同步 Device-to-Host 复制
- 异步 Host-to-Device 复制
- 异步 Device-to-Host 复制
- 同步与异步复制边界检查
- 双 DMA channel 支持
- DMA 故障传播与恢复
- Host memory registration
- Registered memory 的 zero-copy 建模

### 3.4 Stream 与 Event

实现以下能力：

- Stream 创建、同步和销毁
- Stream 内命令 FIFO 顺序
- Stream 间独立命令队列
- 异步错误归属与清除
- Event 创建、记录、查询、同步和销毁
- Event generation 管理
- Event virtual cycle 记录
- Event elapsed-cycle 计算
- 已销毁或伪造 Handle 的安全拒绝
- Stream/Event 生命周期与并发访问保护

异步命令产生的错误保存在其来源 Stream 中，并由相应的 `aecStreamSync` 或等价同步操作返回，避免错误被无关 Stream 或资源管理操作错误消费。

### 3.5 Kernel Launch

Kernel 执行遵循规定的设备执行路径：

1. 调用 `aecDeviceResolveKernel` 解析官方冻结 Kernel Image；
2. 按照设备 ABI 对参数进行小端、紧密排列序列化；
3. 构造 `AEC_DEVICE_OP_ISA_LAUNCH` 命令；
4. 通过 `aecDeviceSubmit` 提交到官方设备库；
5. 根据设备 completion 返回执行结果或错误。

本实现不修改官方冻结 Image，也不使用 Host 代码替代设备 Kernel 计算。

## 4. 计算库支持

### 4.1 GEMM

支持以下数据类型：

- FP4
- FP8 E4M3
- FP8 E5M2
- FP16
- BF16
- FP32
- FP64
- INT4
- INT8
- INT32

相关实现包括：

- GEMM 参数合法性检查
- 最大合法矩阵尺寸检查
- 设备地址范围检查
- 输入与输出区间 overlap 检查
- packed INT4/FP4 数据格式检查
- 小端 Kernel 参数序列化
- 官方冻结 GEMM Image 调用
- 精确整数 GEMM 结果
- 规定的浮点舍入及特殊值行为

### 4.2 向量运算

支持以下 FP32 运算：

- AXPY
- DOT
- NRM2

支持的向量长度范围为：

```text
[1, 1,048,576]
```

DOT 与 NRM2 的完整合法长度直接提交给官方更新后的冻结 Kernel Image 执行，不进行 Host/CPU 代算，也不在 Runtime 中改变规范要求的 FP32 累加顺序。

## 5. ABI 与数值合规

本实现遵守以下约束：

- Runtime ABI 2
- AEC ISA 2 / Profile 1
- C ABI 导出接口
- 设备参数采用 little-endian 编码
- 参数块紧密排列
- 不依赖 C/C++ struct 的 native padding
- 浮点行为遵循赛题数值规范
- NaN 使用规定的 canonical 表示
- 支持 Infinity、subnormal 和 signed zero
- 低精度类型采用规定的 round-to-nearest-even 行为
- packed 类型的低索引元素位于低 nibble
- INT4 计算输出为 little-endian INT32
- 整数结果遵循规定的精确计算或饱和语义
- DOT 按规范要求的 FP32 顺序累加
- NRM2 按规范要求计算并返回 FP32 结果

## 6. 确定性与故障恢复

Runtime 为设备命令分配确定性的单调 sequence，并正确维护：

- submitted commands
- completed commands
- retired instructions
- virtual cycles
- DMA command statistics
- Kernel command statistics
- zero-copy command statistics

同时支持：

- DMA fault injection
- Kernel fault injection
- 故障后的 Stream 恢复
- 故障后的内存与 Host registration 生命周期恢复
- 同步过程中继续排空其他 Stream
- 避免单个 Stream 错误阻止其他 Stream 完成

## 7. 并发与生命周期安全

Runtime 使用同步机制保护以下全局和对象状态：

- 设备内存管理
- Stream/Event Handle registry
- Stream 命令队列
- Event generation
- Host memory registrations
- Runtime statistics
- 命令 sequence

已销毁的 Stream/Event 对象作为可识别的失效 Handle 处理，从而能够安全区分：

- 当前有效 Handle
- 已销毁 Handle
- 从未创建的伪造 Handle

实现同时处理以下生命周期关系：

- 异步复制完成前调用 `aecFree`
- 异步复制完成前调用 `aecHostUnregister`
- Event record 后销毁 Stream
- Event 与 Stream 并发操作
- Stream destroy 与命令 enqueue 竞争
- 故障发生后的资源释放和重试

## 8. Agent 说明

### 8.1 DMA Agent

运行方式：

```bash
echo '<request-json>' | python3 agents/dma_agent.py
```

输入示例：

```json
{
  "case_id": "dma-example",
  "direction": "h2d",
  "bytes": 65536,
  "alignment": 64,
  "registered": true,
  "concurrency": 2
}
```

输出示例：

```json
{
  "channel": 0,
  "chunk_bytes": 1048576,
  "queue_depth": 2,
  "use_zero_copy": true
}
```

Agent 仅向标准输出写出一个 JSON 对象，不输出额外诊断文本。

### 8.2 Kernel Agent

运行方式：

```bash
echo '<request-json>' | python3 agents/kernel_agent.py
```

输入必须包含官方评分协议规定的候选 Kernel 列表。示例：

```json
{
  "case_id": "kernel-example",
  "dtype": "fp32",
  "m": 64,
  "n": 64,
  "k": 64,
  "alignment": 64,
  "workspace": 0,
  "candidates": [
    {
      "id": 101,
      "variant": 1,
      "divisibility": 1,
      "alignment": 1,
      "workspace": 0
    },
    {
      "id": 102,
      "variant": 2,
      "divisibility": 4,
      "alignment": 1,
      "workspace": 0
    },
    {
      "id": 103,
      "variant": 3,
      "divisibility": 8,
      "alignment": 16,
      "workspace": 0
    }
  ]
}
```

输出示例：

```json
{
  "kernel_id": 103
}
```

Kernel Agent 只从官方提供的候选列表中选择满足以下条件的合法 Kernel：

- 数据类型兼容
- M/N/K 整除条件满足
- 地址对齐条件满足
- Workspace 需求不超过限制
- 优先选择预计性能更高的合法变体

Agent 仅向标准输出写出一个 JSON 对象。

## 9. 运行依赖

`libaec.so` 动态依赖官方提供的：

```text
libaec_device.so
```

提交中未包含该文件。验证时需要确保动态链接器可以找到官方设备库，例如：

```bash
export LD_LIBRARY_PATH="<starter-kit>/lib:${LD_LIBRARY_PATH}"
```

随后可检查依赖：

```bash
ldd <submission>/libaec.so
```

预期应能正确解析：

```text
libaec_device.so
libstdc++.so.6
libgcc_s.so.1
libc.so.6
libm.so.6
```

## 10. 验证方式

假设：

```bash
ROOT=<C2 starter-kit 路径>
SUBMIT=<本提交目录路径>
```

运行完整测试：

```bash
cd "$ROOT"

LD_LIBRARY_PATH="$ROOT/lib" \
python3 cases/run_all.py \
  --submission "$SUBMIT"
```

运行公开评分：

```bash
LD_LIBRARY_PATH="$ROOT/lib" \
python3 grader/public_grade.py \
  --submission "$SUBMIT" \
  --profile public
```

单独验证 Agent：

```bash
python3 cases/test_r401.py --submission "$SUBMIT"
python3 cases/test_r402.py --submission "$SUBMIT"
```

## 11. 本地公开测试结果

使用官方更新后的官方 `libaec_device.so` 验证：

```text
16/16 cases passed
AEC score: 88.000000/100
Level: Good
```

公开测试状态：

| 测试项 | 状态 |
|---|---|
| R101 Device metadata and TLS errors | PASS |
| R102 Allocation, reuse, OOM and double-free | PASS |
| R103 Synchronous copy and bounds | PASS |
| R104 Registered vector-add launch | PASS |
| R105 Stream FIFO and asynchronous copy | PASS |
| R106 Event generation and async errors | PASS |
| R201 FP32 and INT32 GEMM | PASS |
| R202 Low-precision and FP64 GEMM | PASS |
| R203 Packed INT4 and INT8 GEMM | PASS |
| R204 AXPY, DOT and NRM2 | PASS |
| R301 Command and completion accounting | PASS |
| R302 Dual DMA and asynchronous bounds | PASS |
| R303 Registration lifecycle and zero-copy | PASS |
| R304 Fault injection and recovery | PASS |
| R401 DMA Agent public correctness | PASS |
| R402 Kernel Agent public legality | PASS |

Agent 的隐藏性能得分由最终评测环境决定，本地公开结果不代表隐藏性能分。

## 12. 提交说明

本提交只包含 C2 交付所需文件：

- `libaec.so`
- `agents/dma_agent.py`
- `agents/kernel_agent.py`
- `README.md`

未包含：

- 官方 `libaec_device.so`
- Grader 或测试用例
- 官方冻结 Kernel Image 的修改版本
- 临时测试文件
- 示例二进制文件
- 编译缓存或中间文件