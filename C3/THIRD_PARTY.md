# Third-Party Software and Development Assistance

本提交不包含第三方项目源码或预编译模型。运行时使用评测环境预装的以下软件包：

| 软件 | 用途 | 许可证 |
|---|---|---|
| Python | 运行环境 | PSF License |
| NumPy | `.npy` 输入输出与 host 数值处理 | BSD-3-Clause |
| ONNX | 模型、图结构及 external-data 解析 | Apache-2.0 |
| ONNX Runtime | 普通 ONNX 模型 GPU/CPU 执行 | MIT |
| CuPy | BigFormer GPU 算子与 CUDA 内存/Stream 管理 | MIT |

所有依赖均由官方评测环境提供，提交代码不会联网下载软件或模型。

开发过程中使用了 OpenAI Codex 辅助代码审查、测试设计、性能实验与文档整理。
最终实现由参赛者审阅、测试并负责维护；未复制其他参赛队伍的代码，也未根据测试
文件名、输入哈希或固定答案生成输出。
