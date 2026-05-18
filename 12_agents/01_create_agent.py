"""
01_create_agent.py
LangChain Agents - 基础 Agent 创建与调用

Agent = LLM + Tools 的自动化循环。
Agent 核心循环示意图：
  https://raw.githubusercontent.com/langchain-ai/docs/main/src/oss/images/core_agent_loop.png

11_tools 模块我们手写了 tool calling 循环（bind_tools → tool_calls → ToolMessage → 循环），
现在用 create_agent 一行代码搞定这个循环。

本文件覆盖：
  - 最简 Agent 创建与调用
  - ReAct 循环：从消息链理解 Agent 的运行机制
  - system_prompt 设定角色
  - 多轮对话

ReAct 与 LangChain 的演变：
  - ReAct（Reasoning + Acting）最早由 Yao et al. 2022 提出（arXiv:2210.03629）
  - LangChain 早期提供 create_react_agent，需手写 prompt 模板和 StateGraph
  - LangGraph 推出后，create_react_agent(langgraph) 用 StateGraph 封装了 ReAct 循环
  - LangChain 0.3+ 统一为 create_agent，内部自动构建 ReAct StateGraph
  - 当前推荐：直接用 create_agent，不再需要 create_react_agent

参考文档：
  - Agents 概述: https://docs.langchain.com/oss/python/langchain/agents
  - create_agent API: https://reference.langchain.com/python/langchain/agents/factory/create_agent
  - LangGraph Agent 核心: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/
  - ReAct 论文: https://arxiv.org/abs/2210.03629

安装：
  pip install langchain langgraph langchain-openai langchain-community python-dotenv
"""

import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


# ========================= 模型配置 =========================

model = ChatOpenAI(
    model="qwen3.6-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
)


# ========================= 自定义工具 =========================

@tool
def add(a: int, b: int) -> int:
    """
    计算两个整数的和
    """
    print(f"工具add正在执行:计算->{a} + {b}")
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积"""
    print(f"工具multiply正在执行:计算->{a} * {b}")
    return a * b


# ========================= 内置工具 =========================

from langchain_community.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()


# ============================================================
# 演示 1：最简单的 Agent
# ============================================================

def demo_basic_agent():
    """
    create_agent 的最简用法。

    调用方式：
      agent = create_agent(model, tools=[...])
      result = agent.invoke({"messages": [HumanMessage(content="...")]})

    invoke(input, config=None) 参数说明：
      input: dict — 第一个参数，输入状态
        {"messages": [BaseMessage, ...]}
        - messages 是唯一必须的字段（input_schema 只有这一个属性）
        - 最简单就传一条 HumanMessage
        - 多轮对话时传完整历史（见 demo 4）
        - schema 来源：agent.input_schema.model_json_schema()

      config: dict | None — 第二个参数，运行时配置（可选）
        {"configurable": {"thread_id": "xxx"}}  — 会话 ID（配合 checkpointer）
        详见 03_agent_checkpointer.py

    invoke 返回值结构：
      {"messages": [BaseMessage, ...]}
      - messages：完整的消息链（HumanMessage → AIMessage → ToolMessage → ... → 最终 AIMessage）
      - 最后一条消息就是最终回复：result["messages"][-1]
      - schema 来源：agent.output_schema.model_json_schema()

    对比 11_tools 的手动循环：
      手动：model.bind_tools → invoke → 取 tool_calls → 执行 → ToolMessage → 再 invoke → ...
      Agent：一行 create_agent，内部自动完成上述循环
    """
    print(">>> 演示 1：最简单的 Agent")
    print("-" * 50)

    #TODO
    agent = create_agent(model, tools=[add, multiply])
    result = agent.invoke({
        "messages": [HumanMessage(content="给我算一下123+34*12等于多少")]#理论上所有上下文历史都可以塞到这里
    })
    final_message = result["messages"][-1]
    print(f"模型回复：{final_message}")
    print()

    # ---- 回头看 agent 的输入输出结构 ----
    # 输入：{"messages": [BaseMessage, ...]}，核心就一个字段
    # 输出：{"messages": [...], "structured_response": ...}
    # 完整 schema 查看：agent.input_schema.model_json_schema() / agent.output_schema.model_json_schema()
    print("agent 的输入/输出 schema:")
    out_schema = agent.output_schema.model_json_schema()
    print(f"  输出字段: {list(out_schema['properties'].keys())}")
    # 输入 schema 用 $ref 引用，解析 $defs 里的 InputSchema
    in_schema = agent.input_schema.model_json_schema()
    input_defs = in_schema.get("$defs", {})
    input_schema_def = input_defs.get("InputSchema", {})
    print(f"  输入字段: {list(input_schema_def.get('properties', {}).keys())}")
    print()


# ============================================================
# 演示 2：查看 Agent 内部的完整消息链
# ============================================================

def demo_react_loop():
    """
    ReAct 模式：Agent 的核心运行机制
    ─────────────────────────────────────
    ReAct（Reasoning + Acting）是一种让 LLM 交替进行"思考"和"行动"的模式：
    Agents 概述: https://docs.langchain.com/oss/python/langchain/agents

      Reasoning（思考）：模型分析问题，决定下一步做什么
      Acting（行动）：调用工具获取信息
      Observation（观察）：工具返回结果
      → 循环，直到模型认为可以给出最终回答

    每一轮 ReAct 循环在消息链中的体现：
      AIMessage(tool_calls=[...])  ← 思考：决定调什么工具、传什么参数
      ToolMessage(content=...)     ← 观察：工具执行结果
      → 如果模型还需要更多信息，继续下一轮；否则给出最终回答

    退出条件：AIMessage 没有 tool_calls → 模型认为信息足够，直接回答

    create_agent 内部就是自动执行这个 ReAct 循环。
    在 11/03 里我们手动写了这个循环（invoke → tool_calls → ToolMessage → invoke），
    create_agent 把它封装成一个 StateGraph：model 节点和 tools 节点循环执行。

    参考文档：
      - ReAct 论文: https://arxiv.org/abs/2210.03629
      - LangGraph Agent 核心: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/
    """
    print(">>> 演示 2：ReAct 循环 —— 从消息链看 Agent 在干什么")
    print("-" * 50)

    agent = create_agent(model, tools=[add, multiply])

    result = agent.invoke({
        "messages": [HumanMessage(content="请先计算 3 加 5，然后把结果乘以 2")]
    })

    # 拆解消息链，展示 ReAct 循环
    messages = result["messages"]
    print(f"问题: 请先计算 3 加 5，然后把结果乘以 2\n")

    round_num = 0
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__

        if isinstance(msg, HumanMessage):
            print(f"[{i}] {msg_type}: {msg.content}")

        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                round_num += 1
                print(f"\n--- ReAct 第 {round_num} 轮 ---")
                tool_names = [tc['name'] for tc in msg.tool_calls]
                print(f"[{i}] AIMessage → 思考: 需要调 {', '.join(tool_names)}")
                for tc in msg.tool_calls:
                    print(f"     tool_call: {tc['name']}({tc['args']})")
            else:
                print(f"\n--- ReAct 循环结束 ---")
                print(f"[{i}] AIMessage → 最终回答: {msg.content}")

        elif hasattr(msg, 'tool_call_id'):
            print(f"[{i}] ToolMessage → 观察: {msg.content}")

    print(f"\n共 {round_num} 轮 ReAct 循环")
    print()


# ============================================================
# 演示 3：system_prompt 设定角色
# ============================================================

def demo_system_prompt():
    """
    system_prompt 控制 Agent 的行为方式和角色定位。
    可以是 str 或 SystemMessage。
    不指定 system_prompt 时，Agent 从消息上下文推断任务。
    """
    print(">>> 演示 3：system_prompt 设定角色")
    print("-" * 50)

    # 方式 1：字符串
    agent1 = create_agent(
        model,
        tools=[add, multiply, search],
        system_prompt="你是一个简洁的数学助手，只回答计算结果，不要废话。"
    )
    result1 = agent1.invoke({
        "messages": [HumanMessage(content="123 乘以 456 等于多少？")]
    })
    print(f"[str] {result1['messages'][-1].content}")

    # 方式 2：SystemMessage（更灵活，支持多段内容、cache_control 等）
    agent2 = create_agent(
        model,
        tools=[search],
        system_prompt=SystemMessage(content=(
            "你是一个搜索助手。\n"
            "规则：只回答搜索相关的问题，其他问题拒绝回答。"
        ))
    )
    result2 = agent2.invoke({
        "messages": [HumanMessage(content="今天吃什么？")]
    })
    print(f"[SystemMessage] {result2['messages'][-1].content}")
    print()


# ============================================================
# 演示 4：多轮对话
# ============================================================

def demo_multi_turn():
    """
    Agent 支持多轮对话，只需传入之前的消息历史。
    这和普通 Chat 模型一样，Agent 会基于完整上下文做决策。
    """
    print(">>> 演示 4：多轮对话")
    print("-" * 50)

    agent = create_agent(model, tools=[add, multiply, search])

    # 第一轮
    result1 = agent.invoke({
        "messages": [HumanMessage(content="帮我算一下 12 乘以 8")]
    })
    print(f"第一轮: {result1['messages'][-1].content}")

    # 第二轮：传入之前的消息历史
    result2 = agent.invoke({
        "messages": result1["messages"] + [HumanMessage(content="再加上刚才结果的 3 倍")]
    })
    print(f"第二轮: {result2['messages'][-1].content}")
    print()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("错误：请设置 DASHSCOPE_API_KEY 环境变量")
        exit(1)

    demo_basic_agent()
    demo_react_loop()
    demo_system_prompt()
    demo_multi_turn()
