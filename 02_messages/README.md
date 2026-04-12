# 02_messages - Messages 组件教学案例

Chat Models 的输入不是纯文本，而是"消息"（Message）。
理解 Message 类型和使用方式是掌握 LangChain 的基础。

## 教学路径

| 文件 | 内容 |
|------|------|
| `01_message_types.py` | 四种消息类型详解 + ToolMessage 完整流程 |
| `02_message_formats.py` | 四种写法对比 + 多轮对话原理 |

## 四种消息类型

| 类型 | 作用 | 示例 |
|------|------|------|
| `SystemMessage` | 设定 AI 角色和行为 | "你是一个翻译助手" |
| `HumanMessage` | 用户输入 | "I love programming." |
| `AIMessage` | 模型回复 | "我爱编程。" |
| `ToolMessage` | 工具执行结果 | "北京: 25°C, 晴" |

## 01_message_types.py

### 基本消息流程

```
SystemMessage → HumanMessage → AIMessage
     ↓               ↓              ↓
  设定角色         用户提问       模型回复
```

### ToolMessage 完整流程（重点）

```
步骤1: 用户提问 → "北京今天天气怎么样？"
步骤2: AI 返回 tool_calls → [{"name": "get_weather", "args": {"city": "北京"}}]
步骤3: 开发者执行工具 → ToolMessage(content="北京: 25°C, 晴", tool_call_id="...")
步骤4: AI 基于工具结果回复 → "北京今天天气不错，25度，晴天。"
```

## 02_message_formats.py

### 四种写法

| 写法 | 示例 | 推荐度 |
|------|------|--------|
| Message 对象 | `SystemMessage(content="...")` | 复杂场景 |
| 元组列表 | `("system", "...")` | ⭐ 日常推荐 |
| 字典列表 | `{"role": "system", "content": "..."}` | 从 API 文档复制 |
| 纯字符串 | `"你好"` | 简单调用 |

### 多轮对话

核心原理：**AI 没有记忆**。每轮调用都需要把历史消息一起传进去。
不传历史 → AI 不知道之前说了什么；传历史 → AI 能看到完整上下文。

## 运行

```bash
# 需要 ZHIPUAI_API_KEY
python 02_messages/01_message_types.py
python 02_messages/02_message_formats.py
```

# LangChain 消息类图关系

基于 LangChain 官方源码（`langchain_core/messages/`），完整展示消息类型的继承关系。

> 源码: <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/messages>

## ASCII 类图

```
langchain_core.load.serializable
│
└── BaseMessage                          # 消息基类 (抽象)
    │   属性: content, type, additional_kwargs, response_metadata, id
    │
    ├── BaseMessageChunk                 # 流式分片基类 (支持 + 拼接)
    │   │
    │   ├── SystemMessageChunk
    │   ├── HumanMessageChunk
    │   ├── AIMessageChunk
    │   └── ToolMessageChunk
    │
    ├── SystemMessage                    # type = "system"   系统指令
    ├── HumanMessage                     # type = "human"    用户输入
    ├── AIMessage                        # type = "ai"       模型输出 (含 tool_calls)
    │
    └── ToolMessage                      # type = "tool"     工具执行结果
```

## 继承关系详解

| 子类            | 父类          | type 值    | 说明                                   |
| --------------- | ------------- | ---------- | -------------------------------------- |
| `SystemMessage` | `BaseMessage` | `"system"` | 系统指令，设置模型行为                 |
| `HumanMessage`  | `BaseMessage` | `"human"`  | 用户输入                               |
| `AIMessage`     | `BaseMessage` | `"ai"`     | 模型输出，独有 `tool_calls` 属性       |
| `ToolMessage`   | `BaseMessage` | `"tool"`   | 工具执行结果，独有 `tool_call_id` 属性 |

## Chunk 变体（流式）

每个消息类型都有对应的 Chunk 版本，用于流式输出（`stream()` / `astream()`）：

| 完整消息        | Chunk 变体           | 继承关系                                  |
| --------------- | -------------------- | ----------------------------------------- |
| `SystemMessage` | `SystemMessageChunk` | 继承 `SystemMessage` + `BaseMessageChunk` |
| `HumanMessage`  | `HumanMessageChunk`  | 继承 `HumanMessage` + `BaseMessageChunk`  |
| `AIMessage`     | `AIMessageChunk`     | 继承 `AIMessage` + `BaseMessageChunk`     |
| `ToolMessage`   | `ToolMessageChunk`   | 继承 `ToolMessage` + `BaseMessageChunk`   |

Chunk 的核心特性：支持 `+` 运算符拼接多个分片。

## Tool Calling 流程中的类协作

```
┌─────────────┐     tool_calls      ┌─────────────────┐
│  AIMessage   │ ──────────────────> │  ToolCall(id,   │
│  (模型决定   │                     │    name, args)  │
│   调用工具)  │                     └────────┬────────┘
└─────────────┘                              │
                                             │ 执行工具
                                             v
                                      ┌─────────────┐     tool_call_id
                                      │ ToolMessage  │ ──────────────>
                                      │ (工具返回    │   关联对应的
                                      │  执行结果)   │   ToolCall
                                      └─────────────┘
```

关键点：

- `AIMessage.tool_calls`: 列表，每个元素包含 `id`、`name`、`args`
- `ToolMessage.tool_call_id`: 字符串，与 `tool_calls[].id` 对应
- 一个 `AIMessage` 可以包含多个 `tool_calls`（并行调用）
- 每个 `ToolMessage` 通过 `tool_call_id` 关联到对应的调用

## BaseMessage 核心属性

```python
class BaseMessage(Serializable):
    content: str | list[str | dict]    # 消息内容
    type: str                          # 消息类型标识
    additional_kwargs: dict            # 扩展字段 (如 tool_calls)
    response_metadata: dict            # 响应元数据 (如 token 用量)
    id: str                            # 消息唯一 ID
```

a 

## 导入方式

```python
from langchain_core.messages import (
    BaseMessage,
    BaseMessageChunk,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    # Chunk 变体
    HumanMessageChunk,
    AIMessageChunk,
    SystemMessageChunk,
    ToolMessageChunk,
)
```
