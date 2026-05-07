"""
01_message_types.py
LangChain Messages - 四种消息类型

Chat Models 的核心是"消息"（Message），而不是纯文本。
LangChain 定义了四种标准消息类型，对应对话中的不同角色：

  SystemMessage  → 设定 AI 的角色和行为（你是什么人）
  HumanMessage   → 用户的输入（用户说了什么）
  AIMessage      → 模型的回复（AI 回了什么）
  ToolMessage    → 工具执行结果（工具返回了什么）

参考文档：
  - Messages 概念: https://docs.langchain.com/oss/python/langchain/messages

安装：
  pip install langchain-community zhipuai
"""

import os

# ============================================================
# 公共：模型实例（单独抽出，方便所有 demo 复用）
# ============================================================

def get_llm(temperature=0):
    """获取智谱 GLM 模型实例"""
    from langchain_deepseek import ChatDeepSeek
    return ChatDeepSeek(
        model="deepseek-chat",
        temperature=temperature,
    )


# ============================================================
# 演示 1：SystemMessage / HumanMessage / AIMessage 基本用法
# ============================================================

def demo_basic_messages():
    """
    三种基本消息类型：System → Human → AI

    对话流程：
      SystemMessage（设定角色）
      → HumanMessage（用户提问）
      → AIMessage（模型回复）
    """
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    llm = get_llm()
    messages = [
        SystemMessage(content="你是一个ai翻译助手，英译中，只翻译不解释"),
        HumanMessage(content="I love programming."),

    ]
    response = llm.invoke(messages)
    print(response)
    print(response.content)

# ============================================================
# 演示 2：ToolMessage 完整流程
# ============================================================

def demo_tool_message():
    """
    ToolMessage 的完整流程（AI 调用工具 → 工具返回结果 → AI 继续回复）

    步骤：
      1. 用户提问
      2. AI 返回 tool_calls（决定调用哪个工具）
      3. 开发者执行工具，构造 ToolMessage
      4. 将 ToolMessage 发回给 AI，AI 基于工具结果继续回复

    这里演示一个"获取天气"的工具调用流程。
    """
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

    llm = get_llm(temperature=0)

    # 定义工具 schema（告诉模型有哪些工具可用）
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称",
                        }
                    },
                    "required": ["city"],
                },
                "additionalProperties": False
            },
        }
    ]

    # 绑定工具到模型
    llm_with_tools = llm.bind_tools(tools)

    print("=== ToolMessage 完整流程 ===\n")

    # 步骤 1：用户提问
    user_msg = HumanMessage(content="北京今天天气怎么样？")
    print(f"步骤1 - 用户提问: {user_msg.content}")

    # 步骤 2：模型返回 tool_calls
    ai_response = llm_with_tools.invoke([user_msg])
    print(f"\n步骤2 - AI 返回:")
    print(f"  AI 内容: {ai_response.content}")  # 可能为空
    print(f"  tool_calls: {ai_response.tool_calls}")

    # 步骤 3：开发者执行工具，构造 ToolMessage
    # （这里模拟执行，实际开发中调用真实 API）
    tool_call = ai_response.tool_calls[0]
    city = tool_call["args"]["city"]
    weather_result = f"""{city}: 今天的天气情况：
                                - 温度：25°C
                                - 天气：晴朗
                                - 湿度：60%
                                - 风速：3级
                                - 空气质量：良好
                                - 建议：适宜户外活动"""  # 模拟工具执行结果

    tool_msg = ToolMessage(
        content=weather_result,
        tool_call_id=tool_call["id"],  # 必须关联到对应的 tool_call
    )
    print(f"\n步骤3 - 工具执行结果:")
    print(f"  ToolMessage.content:     {tool_msg.content}")
    print(f"  ToolMessage.tool_call_id: {tool_msg.tool_call_id}")

    # 步骤 4：将 ToolMessage 发回模型，AI 基于结果回复
    # 注意：这里用不带 tools 的 llm，确保模型直接生成自然语言回复。
    # 如果用 llm_with_tools，部分模型（如 glm-4-flash）可能会再次尝试调用工具。
    # 实际开发中用 LangGraph 的 tool-calling 循环自动处理这个问题，无需手动切换。
    messages = [user_msg, ai_response, tool_msg]
    final_response = llm.invoke(messages)
    print(f"\n步骤4 - AI 最终回复: {final_response.content}")


# ============================================================
# 演示 3：AIMessage 属性详解
# ============================================================

def demo_aimessage_details():
    """
    AIMessage 的详细属性
    """
    llm = get_llm()

    response = llm.invoke("用一句话介绍 Python")

    print("\n=== AIMessage 属性详解 ===")
    print(f"  .content:           {response.content}")
    print(f"  .type:              {response.type}")
    print(f"  .id:                {response.id}")
    print(f"  .response_metadata: {response.response_metadata}")
    print(f"  .usage_metadata:    {response.usage_metadata}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("请设置 DS_API_KEY 环境变量")
        print("  export DS_API_KEY='...'")
        exit(1)

    # 演示 1：基本消息
    demo_basic_messages()

    print("\n" + "=" * 60 + "\n")

    # 演示 2：ToolMessage 完整流程
    demo_tool_message()

    print("\n" + "=" * 60 + "\n")

    # 演示 3：AIMessage 属性
    demo_aimessage_details()
