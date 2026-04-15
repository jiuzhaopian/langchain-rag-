# 06 Chains - LCEL 链式调用

LCEL（LangChain Expression Language）是 LangChain 的核心，用 `|` 管道符组合 Runnable 构建链。
旧版 `LLMChain` 已废弃，现代写法全部用 LCEL。

## 核心概念

```python
# 最基本的三段式链
chain = prompt | llm | parser
result = chain.invoke({"topic": "Python"})
```

`|` 被 Runnable 重载为管道操作：左边的输出自动作为右边的输入。
chain 的类型是 `RunnableSequence`。

## 文件列表

| 文件 | 内容 | 需要 API Key |
|------|------|-------------|
| `01_basic_chain.py` | 两段式/三段式链（chain 类型 RunnableSequence、各环节输入输出类型）、callable 自动入链（不需要手动 RunnableLambda）、batch/stream 自动支持 | 智谱 |
| `02_parallel_branch.py` | RunnableParallel 显式构造 + 管道 dict 两种方式（左侧/右侧均可自动转换）、RunnableBranch 条件分支（条件函数接收 dict） | 智谱 |
| `03_passthrough.py` | RunnablePassthrough 基本透传（恒等函数）、assign 保留原字段并添加新字段、RAG 完整流程图解 | 智谱 |
| `04_chain_principle.py` | 纯 Python 从零实现 MyRunnable + MySequence 模拟 LCEL，理解 `__or__` 运算符重载和链式调用原理 | 无需 Key |

## Runnable 组合模式速查

| 模式 | 写法 | 行为 |
|------|------|------|
| 串行 | `a \| b \| c` | 顺序执行，上个输出是下个输入 |
| 并行 | `{k1: r1, k2: r2}` 在 `\|` 左侧或右侧 | 同一输入分发给多个 Runnable，结果是 dict |
| 条件分支 | `RunnableBranch((cond, r1), default)` | 根据 cond 选择 Runnable，类似 if-elif-else |
| 透传 | `RunnablePassthrough()` | 输入原样输出（恒等函数） |
| 添加字段 | `RunnablePassthrough.assign(k=fn)` | 保留原始所有字段 + 添加新字段 k |

## 关键注意事项

- **dict 自动转换**：`{...}` 在 `|` 左侧或右侧都会自动转为 `RunnableParallel`（`__or__` coerce 右侧，`__ror__` coerce 左侧），直接赋值给变量（`chain = {...}`）不会转换
- **callable 自动包装**：普通函数在 `|` 左侧或右侧都会自动包装为 `RunnableLambda`，不需要手动 `RunnableLambda(fn)`
- **RunnableBranch 条件函数**：接收的是整个输入 dict，不是字符串
- **三段式各环节类型**：`prompt(dict→PromptValue)` → `llm(PromptValue→AIMessage)` → `parser(AIMessage→str/dict/Pydantic)`

## 安装

```bash
pip install langchain-core langchain-community zhipuai
```
