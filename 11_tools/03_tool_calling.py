"""
03_tool_calling.py
LangChain Tools - Tool Calling（工具调用）

Tool Calling 是 LLM 的能力：模型能识别什么时候需要调工具、调哪个、传什么参数。
不是所有模型都支持，需要模型原生支持 function calling / tool use。

核心流程：
  1. 用 model.bind_tools(tools) 把工具注册给模型
  2. 用户提问 → 模型判断是否需要调工具
  3. 需要调 → 返回 AIMessage(tool_calls=[...])
  4. 开发者执行工具，把结果作为 ToolMessage 传回
  5. 模型根据工具结果生成最终回复

参考文档：
  - Tool Calling: https://docs.langchain.com/oss/python/langchain/models#tool-calling
  - Tools 概述: https://docs.langchain.com/oss/python/langchain/tools

安装：
  pip install langchain-openai python-dotenv
"""

import os
import json
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage


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
def get_weather(city: str) -> str:
    """查询指定城市的天气"""
    # 模拟数据
    weather_data = {
        "北京": "晴，25°C",
        "上海": "多云，22°C",
        "深圳": "阵雨，28°C",
    }
    return weather_data.get(city, f"{city}：暂无天气数据")

tools = [add, multiply, get_weather]


# ========================= 模型配置 =========================

def get_llm_with_tools():
    """
    用 ChatOpenAI 连接 DashScope 的 qwen3.6-plus 模型。
    bind_tools() 把工具列表注册到模型，模型就能在回复中生成 tool_calls。
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="qwen3.6-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        temperature=0,
    )
    return llm.bind_tools(tools)


# ============================================================
# 演示 1：模型生成 tool_calls
# ============================================================

def demo_tool_calls_generation():
    """
    当用户问题需要调用工具时，模型不会直接回答，
    而是返回 AIMessage，其中包含 tool_calls 字段。
    """
    llm = get_llm_with_tools()

    print("=== 模型生成 tool_calls ===")

    # 需要调工具的问题
    ai_msg = llm.invoke("3加5等于多少？")

    print(f"问题: 3加5等于多少？")
    print(f"AIMessage.content: '{ai_msg.content}'")  # 通常为空
    print(f"AIMessage.tool_calls: {ai_msg.tool_calls}")
    # tool_calls 结构: [{'name': 'add', 'args': {'a': 3, 'b': 5}, 'id': 'xxx', 'type': 'tool_call'}]
    print()

    # 不需要调工具的问题
    ai_msg2 = llm.invoke("你好，请介绍一下你自己")
    print(f"问题: 你好，请介绍一下你自己")
    print(f"AIMessage.content: {ai_msg2.content[:50]}...")
    print(f"AIMessage.tool_calls: {ai_msg2.tool_calls}")
    print()


# ============================================================
# 演示 2：手动执行工具调用（理解底层流程）
# ============================================================

def demo_manual_tool_execution():
    """
    完整展示 tool calling 的底层流程：
    1. 用户消息 → 2. 模型返回 tool_calls → 3. 执行工具 → 4. 回传结果 → 5. 最终回复

    理解这个过程是理解 Agent 的基础。Agent 只是把这个流程自动化了。
    """
    llm = get_llm_with_tools()

    print("=== 手动执行工具调用（底层流程） ===")

    # Step 1: 用户提问
    user_msg = "北京的天气怎么样？"
    print(f"Step 1 - 用户: {user_msg}")

    # Step 2: 模型判断需要调工具
    ai_msg = llm.invoke(user_msg)
    print(f"Step 2 - 模型返回 tool_calls: {ai_msg.tool_calls}")

    if not ai_msg.tool_calls:
        print("模型认为不需要调工具，直接回复。")
        return

    # Step 3: 执行工具
    tool_call = ai_msg.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    # 根据名字找到工具并执行
    tool_map = {t.name: t for t in tools}
    selected_tool = tool_map[tool_name]
    tool_result = selected_tool.invoke(tool_args)
    print(f"Step 3 - 执行 {tool_name}({tool_args}) = {tool_result}")

    # Step 4: 把工具结果作为 ToolMessage 传回模型
    tool_message = ToolMessage(
        content=str(tool_result),
        tool_call_id=tool_call["id"],  # 必须和 tool_calls 里的 id 对应
    )
    print(f"Step 4 - ToolMessage: content='{tool_message.content}', id='{tool_message.tool_call_id}'")

    # Step 5: 模型根据工具结果生成最终回复
    final_msg = llm.invoke([ai_msg, tool_message])
    print(f"Step 5 - 模型最终回复: {final_msg.content}")
    print()


# ============================================================
# 演示 3：多次工具调用
# ============================================================

def demo_multi_tool_calls():
    """
    模型可能一次返回多个 tool_calls（并行调用），
    也可能在多轮对话中多次调用工具（串行调用）。

    这里演示并行调用：一个问题触发多个工具。
    """
    llm = get_llm_with_tools()

    print("=== 多次工具调用 ===")

    # 一个问题可能触发多个工具
    ai_msg = llm.invoke("3加5等于多少？再算一下4乘6等于多少？")
    print(f"问题: 3加5等于多少？再算一下4乘6等于多少？")
    print(f"tool_calls 数量: {len(ai_msg.tool_calls)}")

    for i, tc in enumerate(ai_msg.tool_calls):
        print(f"  tool_call[{i}]: {tc['name']}({tc['args']})")

    # 执行所有工具调用
    tool_map = {t.name: t for t in tools}
    tool_messages = []
    for tc in ai_msg.tool_calls:
        result = tool_map[tc["name"]].invoke(tc["args"])
        tool_messages.append(ToolMessage(
            content=str(result),
            tool_call_id=tc["id"],
        ))
        print(f"  执行 {tc['name']} → {result}")

    # 传回所有结果
    final_msg = llm.invoke([ai_msg] + tool_messages)
    print(f"最终回复: {final_msg.content}")
    print()


# ============================================================
# 演示 4：tool_choice 控制工具调用行为
# ============================================================

def demo_tool_choice():
    """
    bind_tools() 支持 tool_choice 参数控制模型行为：
      - "auto": 自动决定是否调工具（默认）
      - "none": 禁止调工具

    对比同一个问题在 auto 和 none 下的行为差异。

    注意："any"（强制调工具）和指定具体工具（如 {"type": "tool", "name": "xxx"}）
    不是所有 provider 都支持，使用前需查文档确认。
    """
    from langchain_openai import ChatOpenAI

    llm_base = ChatOpenAI(
        model="qwen3.6-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        temperature=0,
    )

    print("=== tool_choice 控制 ===")
    question = "北京天气怎么样？"
    print(f"同一个问题: {question}\n")

    # auto（默认）——模型自己判断，这个问题会调 get_weather
    llm_auto = llm_base.bind_tools(tools, tool_choice="auto")
    msg_auto = llm_auto.invoke(question)
    print(f"tool_choice='auto':")
    if msg_auto.tool_calls:
        print(f"  → 调用了工具: {[(tc['name'], tc['args']) for tc in msg_auto.tool_calls]}")
    else:
        print(f"  → 直接回复: {msg_auto.content[:50]}")

    # none——禁止调工具，同样的问题但模型只能直接回答
    llm_no_tool = llm_base.bind_tools(tools, tool_choice="none")
    msg_none = llm_no_tool.invoke(question)
    print(f"tool_choice='none':")
    print(f"  → tool_calls={msg_none.tool_calls}")
    print(f"  → 模型只能直接回复: {msg_none.content[:80]}")
    print()


# ============================================================
# 演示 5：tool_call 的数据结构
# ============================================================

def demo_tool_call_structure():
    """
    AIMessage.tool_calls 是一个列表，每个元素是一个 dict：
    {
        "name": str,       # 工具名
        "args": dict,      # 工具参数
        "id": str,         # 调用 ID（用于匹配 ToolMessage）
        "type": "tool_call"  # 固定值
    }

    理解这个结构对调试和理解 Agent 行为很重要。
    """
    llm = get_llm_with_tools()

    print("=== tool_call 数据结构 ===")
    ai_msg = llm.invoke("北京天气怎么样？")

    if ai_msg.tool_calls:
        tc = ai_msg.tool_calls[0]
        print(f"name: {tc['name']} (type: {type(tc['name']).__name__})")
        print(f"args: {tc['args']} (type: {type(tc['args']).__name__})")
        print(f"id: {tc['id']} (type: {type(tc['id']).__name__})")
        print(f"type: {tc['type']}")
    print()

    # AIMessage 还有 invalid_tool_calls 字段
    print(f"invalid_tool_calls: {ai_msg.invalid_tool_calls}")
    print()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("错误：请设置 DASHSCOPE_API_KEY 环境变量")
        print("  export DASHSCOPE_API_KEY=sk-xxx")
        print("  或在 ../.env 文件中配置")
        exit(1)

    demo_tool_calls_generation()
    demo_manual_tool_execution()
    demo_multi_tool_calls()
    demo_tool_choice()
    demo_tool_call_structure()
