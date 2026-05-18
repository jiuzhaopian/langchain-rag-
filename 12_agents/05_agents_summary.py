"""
05_agents_summary.py
LangChain Agents - 模块总结

覆盖 12_agents 模块全部内容的速查表。
写项目时遇到问题翻这个文件。

参考文档：
  - Agents 概述: https://docs.langchain.com/oss/python/langchain/agents
  - Streaming: https://docs.langchain.com/oss/python/langgraph/streaming
  - Persistence: https://docs.langchain.com/oss/python/langgraph/persistence
"""

# ============================================================
# 1. Agent 核心流程
# ============================================================

AGENT_FLOW = """
Agent 核心流程（ReAct 模式）：

  用户输入
    → LLM 推理（需要调工具吗？调哪个？）
    → 调用工具
    → 工具返回结果
    → LLM 再推理（结果够了吗？还需要其他工具吗？）
    → ...循环...
    → LLM 生成最终回复

对比 11_tools：
  11_tools 手写循环：bind_tools → invoke → 取 tool_calls → 执行 → ToolMessage → 再 invoke
  12_agents 一行搞定：create_agent(model, tools=tools)
"""


# ============================================================
# 2. create_agent 参数速查
# ============================================================

CREATE_AGENT_PARAMS = """
create_agent 核心参数（已验证自源码签名）：

  必选：
    model: str | BaseChatModel        -- 模型实例或字符串标识

  常用：
    tools: Sequence[BaseTool | Callable | dict]  -- 工具列表
    system_prompt: str | SystemMessage            -- 角色和行为规则
    middleware: Sequence[AgentMiddleware]          -- 中间件
    checkpointer: Checkpointer                    -- 状态快照（会话记忆）

  进阶：
    response_format: ResponseFormat               -- 结构化输出（JSON 等）
    state_schema: type[AgentState]                -- 自定义状态结构
    name: str                                     -- 多 Agent 节点标识
    store: BaseStore                              -- 长期存储
    cache: BaseCache                              -- 响应缓存
"""


# ============================================================
# 3. Middleware Hook 速查
# ============================================================

MIDDLEWARE_HOOKS = """
Middleware 6 个 Hook 位置：

  ┌─────────────────────────────────────────────────────────┐
  │  before_agent（整个调用前，Node-style）                   │
  │  ┌─────────────────────────────────────────────────────┐│
  │  │  before_model → [wrap_model_call → model_call]      ││
  │  │  → after_model                                        ││
  │  │       ↓ (如果有 tool_calls)                         ││
  │  │  wrap_tool_call → [执行工具] → 返回结果              ││
  │  │       ↓ (继续循环直到无 tool_calls)                  ││
  │  └─────────────────────────────────────────────────────┘│
  │  after_agent（整个调用后，Node-style）                    │
  └─────────────────────────────────────────────────────────┘

  Node-style（节点式）— 在特定执行点运行，适合日志、校验、修改状态
    before_agent(state, runtime) → dict | None
    after_agent(state, runtime)  → dict | None
    before_model(state, runtime) → dict | None
    after_model(state, runtime)  → dict | None

  Wrap-style（包裹式）— 包裹每次调用，适合重试、缓存、转换
    wrap_model_call(request, handler) → response
    wrap_tool_call(request, handler)  → response

内置 Middleware（已验证自源码签名）：
  ToolRetryMiddleware(max_retries=2, backoff_factor=2.0)
      -- 工具失败自动重试，支持指数退避 + jitter
  ModelCallLimitMiddleware(run_limit=None, thread_limit=None, exit_behavior='end')
      -- 限制模型调用次数，run_limit=单次调用上限，thread_limit=单线程上限
  SummarizationMiddleware(model, trigger=None, keep=('messages', 20))
      -- 长对话自动摘要，trigger=触发条件，keep=摘要保留量

多个 Middleware 可组合使用，顺序即执行顺序。
"""


# ============================================================
# 4. Checkpointer 速查
# ============================================================

CHECKPOINTER = """
Checkpointer — 会话记忆与持久化

核心概念：
  Thread：通过 thread_id 标识的会话，同一 thread 共享状态
  Checkpoint：每步执行后的状态快照
  用法：agent.invoke({"messages": [...]}, config={"configurable": {"thread_id": "xxx"}})

持久化方案：
  MemorySaver    -- 内存，开发/测试用，进程重启丢失（内置）
  SqliteSaver    -- SQLite 文件，单机/中小项目
  PostgresSaver  -- PostgreSQL，生产/多实例部署
  Async 版本     -- 异步场景（aio 子模块）

  社区：Redis（langgraph-checkpoint-redis）、MongoDB（langgraph-checkpoint-mongodb）

发现更多方案：
  PyPI:   https://pypi.org/search/?q=langgraph-checkpoint
  GitHub: https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint
  文档:   https://docs.langchain.com/oss/python/langgraph/persistence#checkpointer-options

常见陷阱：
  ❌ 忘记传 thread_id → 无法保存/恢复状态
  ❌ 不同会话用同一个 thread_id → 状态串了
  ✅ config={"configurable": {"thread_id": "unique-per-conversation"}}
"""


# ============================================================
# 5. Streaming 速查
# ============================================================

STREAMING = """
Streaming — 流式输出

来源：https://docs.langchain.com/oss/python/langgraph/streaming

企业实践中唯一需要掌握的模式：messages（打字机效果）。

用法：
  agent.stream(input, stream_mode="messages")

每个元素是 (MessageChunk, metadata) 元组：
  for msg, metadata in agent.stream(input, stream_mode="messages"):
      if msg.content:  # 过滤 content=None 的 chunk（如 qwen 的思考阶段）
          print(msg.content, end="", flush=True)

metadata 关键字段：
  langgraph_node    -- 当前节点名（model/tools）
  langgraph_step    -- 当前执行步骤
  ls_model_name     -- 模型名
  ls_provider       -- 模型提供者

注意事项：
  - 模型需要 streaming=True（ChatOpenAI 参数）才能产出 chunk 级输出
  - chunk 粒度取决于模型的 streaming 实现，不一定是逐 token
  - messages 只产出 LLM 的 token，不含 ToolMessage

全部 7 种 stream_mode（仅作参考，调试用）：
  "values"      -- 每步后完整 state 快照
  "updates"     -- 每步后状态变化（增量）
  "messages"    -- ⭐ LLM token + metadata（产品标配）
  "custom"      -- 节点内 get_stream_writer 自定义
  "checkpoints" -- checkpoint 事件（需 checkpointer）
  "tasks"       -- 任务开始/结束事件
  "debug"       -- 全量调试信息

  组合用法：stream_mode=["updates", "messages"]，建议加 version="v2"
"""


# ============================================================
# 6. 常见陷阱
# ============================================================

COMMON_PITFALLS = """
常见陷阱：

  ❌ 工具 docstring 太模糊 → Agent 不知道什么时候该用
  ✅ docstring 要清晰描述功能、参数含义

  ❌ 期望 Agent 总是调工具 → 有时直接回答更合理
  ✅ Agent 会自行判断是否需要工具

  ❌ 用 pre-bound model（已 bind_tools）+ 结构化输出
  ✅ create_agent 内部自动 bind_tools

  ❌ system_prompt 过长 → 挤占上下文
  ✅ 简洁明确，规则精简

  ❌ 不限制循环次数 → 无限循环烧钱
  ✅ 用 ModelCallLimitMiddleware 或 LangGraph 默认递归限制

  ❌ Checkpointer 忘记传 thread_id → 无法共享记忆
  ✅ config={"configurable": {"thread_id": "xxx"}}

  ❌ stream_mode="messages" 没加 streaming=True → 收不到 token chunk
  ✅ ChatOpenAI(..., streaming=True)

  ❌ 不过滤 content=None → 思考阶段输出空白
  ✅ if msg.content: print(msg.content)
"""


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LangChain Agents 模块总结")
    print("=" * 60)

    print("\n--- 1. Agent 核心流程 ---")
    print(AGENT_FLOW)

    print("--- 2. create_agent 参数速查 ---")
    print(CREATE_AGENT_PARAMS)

    print("--- 3. Middleware Hook 速查 ---")
    print(MIDDLEWARE_HOOKS)

    print("--- 4. Checkpointer 速查 ---")
    print(CHECKPOINTER)

    print("--- 5. Streaming 速查 ---")
    print(STREAMING)

    print("--- 6. 常见陷阱 ---")
    print(COMMON_PITFALLS)
