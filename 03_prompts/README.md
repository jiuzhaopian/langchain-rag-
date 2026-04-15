# 04_prompts - Prompts 组件教学案例

Prompt（提示词）是给模型的"指令"。LangChain 的 Prompt 模板把 prompt 从硬编码变成可复用、可维护的组件。

## 教学路径

| 文件 | 内容 | 重要度 |
|------|------|--------|
| `01_prompt_template.py` | PromptTemplate：纯文本模板，了解即可 | 了解 |
| `02_chat_prompt_template.py` | ChatPromptTemplate：聊天消息模板（重点） | 重点 |
| `03_messages_placeholder.py` | MessagesPlaceholder：插入对话历史 | 重点 |
| `04_few_shot_prompt.py` | Few-Shot Prompting：少样本提示 | 进阶 |

## 核心概念

### 两种模板类型

| 类型 | 输出 | 适用场景 |
|------|------|----------|
| `PromptTemplate` | `StringPromptValue`（纯文本） | 旧版 LLM（已废弃） |
| `ChatPromptTemplate` | `ChatPromptValue`（消息列表） | Chat Model（推荐） |

### ChatPromptTemplate 的消息格式

| 格式 | 示例 | 说明 |
|------|------|------|
| 元组 | `("system", "你是{role}")` | 最常用 |
| Message 对象 | `SystemMessage(content="...")` | 静态消息 |
| 纯字符串 | `"{input}"` | 默认为 human |
| 字典 | `{"role": "human", "content": "..."}` | 从 API 文档复制 |

### MessagesPlaceholder vs 纯文本变量

| 方式 | 结构 | 推荐度 |
|------|------|--------|
| 纯文本变量 | 所有历史拼成一条 human 消息 | 不推荐 |
| `MessagesPlaceholder` | 保留每条消息的原始类型 | 推荐 |

### LCEL 链式调用

Prompt 和 Chat Model 都是 Runnable，可以用 `|` 串联：

```python
chain = prompt | llm
result = chain.invoke({"variable": "value"})
```

## 01_prompt_template.py

纯文本模板，输出 `StringPromptValue`。

关键方法：
- `from_template()` — 自动提取变量名
- `partial()` — 预填充部分变量
- `invoke()` — 填充所有变量
- `to_string()` / `to_messages()` — 转换输出

## 02_chat_prompt_template.py

聊天消息模板，输出 `ChatPromptValue`。支持多种消息格式组合，可嵌入 few-shot 示例。

## 03_messages_placeholder.py

`MessagesPlaceholder("history")` 在模板中预留一个位置，运行时传入一组消息列表。支持 `optional=True`（不传也不报错）。

多轮对话的标准写法：

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "..."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])
```

## 04_few_shot_prompt.py

通过提供"输入-输出"示例引导模型行为。

| 类 | 适用 |
|----|------|
| `FewShotPromptTemplate` | 纯文本 prompt |
| `FewShotChatMessagePromptTemplate` | Chat Model 消息 |

## 运行

```bash
# 01 不需要 API Key
python 04_prompts/01_prompt_template.py

# 02-04 需要 ZHIPUAI_API_KEY
export ZHIPUAI_API_KEY="..."
python 04_prompts/02_chat_prompt_template.py
python 04_prompts/03_messages_placeholder.py
python 04_prompts/04_few_shot_prompt.py
```

## 参考

- Prompt Templates: https://python.langchain.com/docs/how_to/prompt_templates/
- Chat Prompts: https://python.langchain.com/docs/how_to/chat_prompts/
- Few-Shot: https://python.langchain.com/docs/how_to/few_shot_examples/
