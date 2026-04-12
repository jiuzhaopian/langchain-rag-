# 01_chat_models - Chat Models 教学案例

Chat Models 是 LangChain 1.x 的核心模型接口，接收消息列表，返回消息对象。

## 教学路径

按 provider 分步学习，逐步深入：

| 序号 | 文件 | Provider | 要点 |
|------|------|----------|------|
| 1 | `01_openai_compatible.py` | OpenAI 兼容 | DashScope/DeepSeek 的兼容模式用法 |
| 2 | `02_dashscope_native.py` | 通义原生 | ChatTongyi 原生接口 |
| 3 | `03_deepseek.py` | DeepSeek | OpenAI 兼容模式调用 DeepSeek |
| 4 | `04_zhipu.py` | 智谱 | langchain-glm 官方包 |
| 5 | `05_ollama.py` | Ollama | 本地模型，无需 API Key |
| 6 | `06_init_chat_model.py` | 通用 | 统一初始化器 + 动态切换 |
| 7 | `07_stream.py` | - | 流式调用 |
| 8 | `08_batch.py` | - | 批量调用 |

## 核心概念

**所有 Chat Model 共享相同接口：**

```python
response = model.invoke("你好")              # 同步调用
for chunk in model.stream("你好"): ...       # 流式调用
responses = model.batch(["问题1", "问题2"])  # 批量调用
```

**返回值类型：**
- `invoke()` → `AIMessage`（有 `.content`、`.usage_metadata`、`.response_metadata`）
- `stream()` → 迭代 `AIMessageChunk`（每个 chunk 有 `.content`）
- `batch()` → `List[AIMessage]`

## Provider 选择指南

| Provider | 包名 | 特点 | 适用场景 |
|----------|------|------|----------|
| DashScope 兼容 | `langchain-openai` | 用 ChatOpenAI + base_url | 需要快速接入 |
| DashScope 原生 | `dashscope` | ChatTongyi，支持 tool calling | 深度使用通义功能 |
| DeepSeek | `langchain-openai` | 纯 OpenAI 兼容 | DeepSeek R1/V3 |
| 智谱 | `langchain-glm` | 官方包，支持 GLM-4 All Tools | 智谱生态 |
| Ollama | `langchain-ollama` | 本地运行，零成本 | 开发调试、隐私场景 |

## 运行

```bash
# 每个 py 文件可独立运行
python 01_openai_compatible.py
python 05_ollama.py          # Ollama 无需 API Key
python 07_stream.py          # 流式调用
```
