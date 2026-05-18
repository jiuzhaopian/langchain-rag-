"""
02_agent_middleware.py
LangChain Agents - Middleware

Middleware 在 Agent 执行流程中插入自定义逻辑。
通过 hook（钩子）拦截 Agent 的每一步，实现日志、重试、限流、摘要等横切关注点。

Middleware Hook 位置示意图：
  https://raw.githubusercontent.com/langchain-ai/docs/main/src/oss/images/middleware_final.png

两种 hook 风格：
  1. Node-style（节点式）：在特定执行点运行一个函数，适合日志、校验、修改状态
     - before_agent / after_agent：整个调用前后
     - before_model / after_model：每次模型调用前后
     - 函数签名：(state, runtime) → dict | None（返回 dict 会合并到 state）

  2. Wrap-style（包裹式）：包裹每次调用，可以修改请求/响应或拦截异常
     - wrap_model_call：包裹模型调用
     - wrap_tool_call：包裹工具执行
     - 函数签名：(request, handler) → response（必须调用 handler(request) 并返回）

参考文档：
  - Middleware 概述: https://docs.langchain.com/oss/python/langchain/middleware/overview
  - 内置 Middleware: https://docs.langchain.com/oss/python/langchain/middleware/built-in
  - 自定义 Middleware: https://docs.langchain.com/oss/python/langchain/middleware/custom

安装：
  pip install langchain langgraph langchain-openai python-dotenv
"""

import os
import time
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage


# ========================= 模型配置 =========================

model = ChatOpenAI(
    model="qwen3.6-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
)


# ========================= 工具定义 =========================

@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积"""
    return a * b

@tool
def flaky_search(query: str) -> str:
    """模拟一个不稳定的搜索工具，每次 agent 运行的前 2 次调用必失败，第 3 次成功"""
    if not hasattr(flaky_search, "_call_count"):
        flaky_search._call_count = 0
    flaky_search._call_count += 1
    call = flaky_search._call_count
    print(f"  [flaky_search] 第 {call} 次调用...")
    if call <= 2:
        raise ConnectionError(f"搜索服务暂时不可用（第 {call} 次失败）")
    return f"搜索 '{query}' 的结果：\n1. LangChain 发布全新 Agent 中间件系统，支持 6 种生命周期 Hook\n2. OpenAI 推出 GPT-5 Turbo 模型，推理速度提升 3 倍\n3. Anthropic Claude 新增工具调用优化，延迟降低 40%"

tools = [add, multiply, flaky_search]


# ============================================================
# 演示 1：6 种 hook 位置全覆盖（一个案例）
# ============================================================

def demo_all_hooks():
    """
    用一个完整的 Agent 调用，同时挂载所有 6 种 hook，
    观察 hook 的触发顺序和位置。

    Agent 执行流程（1 次工具调用场景）：
      1. before_agent（调用开始）
      2. before_model（第一次模型调用前）
      3. wrap_model_call（包裹第一次模型调用）
      4. after_model（第一次模型返回 tool_calls）
      5. wrap_tool_call（包裹工具执行）
      6. before_model（第二次模型调用前）
      7. wrap_model_call（包裹第二次模型调用）
      8. after_model（第二次模型返回最终回复）
      9. after_agent（调用结束）
    """
    print(">>> 演示 1：6 种 hook 位置全覆盖")
    print("-" * 50)

    from langchain.agents.middleware import (
        before_agent, after_agent,
        before_model, after_model,
        wrap_model_call, wrap_tool_call,
        AgentState, ModelRequest, ModelResponse,
    )
    from langgraph.runtime import Runtime
    from typing import Any, Callable

    step = [0]  # 用列表计数（闭包可变）

    @before_agent
    def on_before_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        step[0] += 1
        print(f"  Step {step[0]}: [before_agent] 调用开始，消息数: {len(state['messages'])}")
        return None

    @after_agent
    def on_after_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        step[0] += 1
        final = state["messages"][-1].content[:60] if hasattr(state["messages"][-1], "content") else ""
        print(f"  Step {step[0]}: [after_agent] 调用结束，最终回复: {final}...")
        return None

    @before_model
    def on_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        step[0] += 1
        print(f"  Step {step[0]}: [before_model] 准备调用模型，当前消息数: {len(state['messages'])}")
        return None

    @after_model
    def on_after_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        step[0] += 1
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            print(f"  Step {step[0]}: [after_model] 模型请求调用工具: {[tc['name'] for tc in last.tool_calls]}")
        else:
            content = last.content[:60] if hasattr(last, "content") else ""
            print(f"  Step {step[0]}: [after_model] 模型最终回复: {content}...")
        return None

    @wrap_model_call
    def on_wrap_model(request: ModelRequest, handler: Callable) -> ModelResponse:
        step[0] += 1
        start = time.time()
        print(f"  Step {step[0]}: [wrap_model_call] 模型调用开始...")
        response = handler(request)
        elapsed = time.time() - start
        print(f"             [wrap_model_call] 模型调用完成，耗时: {elapsed:.2f}s")
        return response

    @wrap_tool_call
    def on_wrap_tool(request, handler: Callable):
        step[0] += 1
        tool_name = getattr(request, 'tool_call', {}).get('name', str(request)) if hasattr(request, 'tool_call') else getattr(request, 'name', str(request))
        print(f"  Step {step[0]}: [wrap_tool_call] 工具调用: {tool_name}")
        response = handler(request)
        result = getattr(response, 'content', str(response))
        print(f"             [wrap_tool_call] 工具返回: {result}")
        return response

    agent = create_agent(
        model,
        tools=[add, multiply],
        middleware=[
            on_before_agent, on_after_agent,
            on_before_model, on_after_model,
            on_wrap_model, on_wrap_tool,
        ],
    )

    result = agent.invoke({"messages": [HumanMessage(content="3 加 5 等于几？")]})
    print(f"\n最终回复: {result['messages'][-1].content}")
    print(f"总计触发 {step[0]} 次 hook")
    print()


# ============================================================
# 演示 2：内置 Middleware —— ToolRetryMiddleware
# ============================================================

def demo_tool_retry():
    """
    ToolRetryMiddleware 自动重试失败的工具调用。
    适合调用外部 API、数据库等不稳定的服务。

    参数：
      - max_retries: 最大重试次数（默认 3）
      - tools: 指定对哪些工具启用重试（None = 全部）
      - backoff_factor: 退避因子
      - initial_delay: 首次重试延迟（秒）
    """
    print(">>> 演示 2：ToolRetryMiddleware —— 工具重试")
    print("-" * 50)

    from langchain.agents.middleware import ToolRetryMiddleware

    agent = create_agent(
        model,
        tools=tools,
        middleware=[
            ToolRetryMiddleware(
                max_retries=3,
                initial_delay=0.1,
                backoff_factor=2,
            )
        ],
        system_prompt="你是搜索助手。对每个问题只搜索一次，拿到结果后直接总结回复，不要重复搜索。",
    )

    # flaky_search 前 2 次必失败，第 3 次成功 —— 能稳定复现重试行为
    # 注意：模型可能对同一工具发起多次调用（比如觉得给的结果信息不够），这里为了演示效果在 system_prompt 加了限制
    #       每次工具调用都会独立触发 ToolRetry 重试。所以总调用次数可能超过 3 次。
    flaky_search._call_count = 0  # 重置计数器
    result = agent.invoke({
        "messages": [HumanMessage(content="帮我搜索一下今天的新闻")]
    })

    for msg in result["messages"]:
        msg_type = type(msg).__name__
        if msg_type == "ToolMessage":
            print(f"  工具结果: {msg.content[:100]}")
        elif msg_type == "AIMessage" and msg.content:
            print(f"  最终回复: {msg.content[:200]}")
    print()


# ============================================================
# 演示 3：内置 Middleware —— ModelCallLimitMiddleware
# ============================================================

def demo_model_call_limit():
    """
    ModelCallLimitMiddleware 限制模型调用次数，防止成本失控。

    参数：
      - thread_limit: 单个 thread 总调用上限
      - run_limit: 单次 invoke 调用上限
      - exit_behavior: 达到限制时的行为（"error" 抛异常 / "end" 提前结束）

    企业实战中这个非常有用，防止 Agent 陷入无限循环烧钱。
    """
    print(">>> 演示 3：ModelCallLimitMiddleware —— 成本控制")
    print("-" * 50)

    from langchain.agents.middleware import ModelCallLimitMiddleware

    # run_limit=2 限制最多 2 次模型调用
    # "先算 123+456，再算结果乘以 2" 正常需要 3 次模型调用（调 add → 调 multiply → 回复）
    # 但 run_limit=2 会在第 2 次模型调用后强制结束，用户能看到任务被截断
    agent = create_agent(
        model,
        tools=[add, multiply],
        middleware=[
            ModelCallLimitMiddleware(
                run_limit=2,
                exit_behavior="end",
            )
        ],
    )

    result = agent.invoke({
        "messages": [HumanMessage(content="先算 123+456，再算结果乘以 2")]
    })
    # 打印完整消息链，看被截断前的中间结果
    for msg in result['messages']:
        msg_type = type(msg).__name__
        if msg_type == "ToolMessage":
            print(f"  工具结果: {msg.content}")
        elif msg_type == "AIMessage" and msg.content:
            print(f"  AI回复: {msg.content[:200]}")
        elif msg_type == "HumanMessage":
            print(f"  用户: {msg.content}")
    print(f"\n  最终消息: {result['messages'][-1].content}")
    print()


# ============================================================
# 演示 4：内置 Middleware —— SummarizationMiddleware
# ============================================================

def demo_summarization():
    """
    SummarizationMiddleware 在对话接近 token 上限时自动摘要历史消息，
    保留最近的消息，压缩较早的上下文。

    参数：
      - model: 用于生成摘要的模型（可用更便宜的模型）
      - trigger: 触发摘要的条件（如 ("tokens", 200)）
      - keep: 摘要后保留的消息数（如 ("messages", 4)）

    摘要流程（本 demo 的执行过程）：
      1. 预填 10 条长消息 + 1 条新问题 = 11 条，约 500 tokens（>> 200 阈值）
      2. Agent 进入模型调用 → SummarizationMiddleware 检测到超阈值
      3. Middleware 内部调模型生成摘要（触发第 1 次 before_model）
      4. 10 条旧消息被压缩为 1 条摘要 + 4 条保留 = 5 条
      5. Agent 正式调模型回答问题（触发第 2 次 before_model）
      6. 最终输出：摘要后的消息 + 回复
    """
    print(">>> 演示 4：SummarizationMiddleware —— 长对话摘要")
    print("-" * 50)
    print("  摘要流程：")
    print("    1. 预填 10 条长消息（~500 tokens），超过 200 tokens 阈值")
    print("    2. SummarizationMiddleware 检测超阈值 → 内部调模型生成摘要")
    print("    3. 10 条旧消息压缩为 1 条摘要 + 保留最近 4 条 = 5 条")
    print("    4. Agent 用摘要后的消息正式回答问题")
    print()

    from langchain.agents.middleware import SummarizationMiddleware, before_model
    from langchain_core.messages import HumanMessage, AIMessage

    call_count = [0]  # 记录模型调用次数

    @before_model
    def track_model_call(state, runtime):
        call_count[0] += 1
        msgs = state.get("messages", [])
        if call_count[0] == 1:
            print(f"  ▶ 第 {call_count[0]} 次模型调用（SummarizationMiddleware 生成摘要）")
        else:
            print(f"  ▶ 第 {call_count[0]} 次模型调用（Agent 正式回答问题）")
        # 检查是否有摘要消息（SummarizationMiddleware 用 HumanMessage 存摘要，内容以 "Here is a summary" 开头）
        has_summary = any(
            isinstance(m, HumanMessage) and 'summary' in m.content[:30].lower()
            for m in msgs
        )
        print(f"    消息数: {len(msgs)}，包含摘要: {'是' if has_summary else '否'}")
        return None

    agent = create_agent(
        model,
        tools=[add, multiply],
        middleware=[
            SummarizationMiddleware(
                model=model,
                trigger=("tokens", 200),
                keep=("messages", 4),
            ),
            track_model_call,
        ],
        system_prompt="你是一个数学助手。简短回答即可。"
    )

    # 预填 10 条长消息
    history = []
    for i in range(1, 6):
        history.append(HumanMessage(content=f"第 {i} 个问题：请详细解释一下数学中关于整数运算的基本概念和性质，包括加法交换律、结合律以及乘法分配律的应用场景"))
        history.append(AIMessage(content=f"第 {i} 个问题的回答：整数运算的基本概念包括加法交换律（a+b=b+a）、结合律（(a+b)+c=a+(b+c)）、乘法分配律（a×(b+c)=a×b+a×c）。这些性质在代数运算、方程求解、几何证明等领域有广泛应用。"))

    print(f"  输入: {len(history)} 条历史消息 + 1 条新问题 = {len(history) + 1} 条（约 {len(history) * 50} tokens）")
    print(f"  阈值: 200 tokens，保留: 最近 4 条消息")
    print()
    print("  --- 开始执行 ---")

    result = agent.invoke({
        "messages": history + [HumanMessage(content="100 加 200 等于多少？")]
    })

    print("  --- 执行完毕 ---")
    print()
    print(f"  最终回复: {result['messages'][-1].content[:100]}")
    print(f"  最终消息数: {len(result['messages'])}（从初始 {len(history) + 1} 条压缩而来）")
    print(f"  模型总调用次数: {call_count[0]}（1 次摘要 + 1 次回答）")
    print()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("错误：请设置 DASHSCOPE_API_KEY 环境变量")
        exit(1)

    demo_all_hooks()
    demo_tool_retry()
    demo_model_call_limit()
    demo_summarization()