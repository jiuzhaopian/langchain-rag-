"""
02_message_formats.py
Messages 写法对比 - 同样的对话，四种写法

LangChain 支持多种方式构造消息，灵活但容易混淆。
本文件展示所有写法并做对比，帮你在不同场景下选对写法。

参考文档：
  - Messages 概念: https://docs.langchain.com/oss/python/langchain/messages

安装：
  pip install langchain-community zhipuai
"""

import os


def get_llm(temperature=0):
    """获取智谱 GLM 模型实例"""
    from langchain_community.chat_models import ChatZhipuAI
    return ChatZhipuAI(
        model="glm-4.7",
        temperature=temperature,
    )


# ============================================================
# 四种写法对比
# ============================================================

def demo_format_1_message_objects():
    """
    写法 1：Message 对象列表

    最显式、最清晰，IDE 有补全提示。
    适合：复杂对话、需要精确控制消息类型的场景
    """
    from langchain.messages import SystemMessage, HumanMessage # 也是从langchain_core.messages导入的
    #TODO
    pass


def demo_format_2_tuples():
    """
    写法 2：元组列表（推荐日常使用）

    简洁，代码量少。Chat Model 自动转换：
      ("system", "...") → SystemMessage
      ("human", "...")  → HumanMessage
      ("ai", "...")     → AIMessage
    """
    #TODO
    pass


def demo_format_3_dict():
    """
    写法 3：字典列表（与 OpenAI API 格式一致）

    方便从 OpenAI API 文档直接复制示例。
    注意：OpenAI 用 "user"，LangChain 元组用 "human"
    """
    #TODO
    pass


def demo_format_4_single_string():
    """
    写法 4：直接传字符串（最简单）

    无法设置 system prompt，无法传多轮对话。
    适合：简单的单轮调用
    """
    response = get_llm().invoke("你只回复一个词：猫")
    print(f"[纯字符串]     {response.content}")


# ============================================================
# 多轮对话：为什么要传历史消息
# ============================================================

def demo_multi_turn():
    """
    多轮对话：AI 没有记忆，需要把历史消息传进去

    问题：AI 每次调用都是无状态的，不知道上一轮说了什么。
    解决：把之前的对话历史（HumanMessage + AIMessage）一起传给模型。

    消息顺序：
      SystemMessage（设定角色，可选）
      → HumanMessage（第1轮用户消息）
      → AIMessage（第1轮AI回复）
      → HumanMessage（第2轮用户消息）
      → AIMessage（第2轮AI回复）
      → HumanMessage（当前用户消息）
    """
    from langchain.messages import SystemMessage, HumanMessage, AIMessage

    llm = get_llm()

    # 不传历史 → AI 不知道之前说了什么
    print("=== 不传历史 ===")
    response = llm.invoke("我刚才说我爱什么？")
    print(f"  AI: {response.content}")  # AI 不知道之前说了 "I love programming."

    # 传历史 → AI 能看到上下文
    print("\n=== 传历史 ===")
    messages = [
        SystemMessage(content="你是一个翻译助手"),
        HumanMessage(content="I love programming."),       # 第1轮：用户
        AIMessage(content="我爱编程。"),                     # 第1轮：AI 回复
        HumanMessage(content="我刚才说我爱什么？"),         # 第2轮：用户追问
    ]
    response = llm.invoke(messages)
    print(f"  AI: {response.content}")  # AI 能看到历史，知道是英语


# ============================================================
# 对比总结
# ============================================================

def demo_comparison_table():
    """写法对比总结"""
    print("""
四种写法对比：

写法            示例                  类型安全   推荐场景
─────────────────────────────────────────────────────────────
Message 对象    SystemMessage(...)    ✅ 最高    复杂对话、精确控制
元组列表        ("system", "...")     ⚠️ 中等    ⭐ 日常推荐
字典列表        {"role": "system",..}  ❌ 最低    从 OpenAI 文档复制
纯字符串        "你好"                ❌ 无      简单单轮调用

关键区别：
  - 元组写法用 "human" / "ai"
  - 字典写法用 "user" / "assistant"（OpenAI 格式）
  - 所有写法最终都被 Chat Model 转为统一的 Message 对象
""")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    if not os.environ.get("ZHIPUAI_API_KEY"):
        print("请设置 ZHIPUAI_API_KEY 环境变量")
        print("  export ZHIPUAI_API_KEY='...'")
        exit(1)

    print("=== Messages 写法对比 ===\n")

    print("--- 1. 四种写法 ---")
    demo_format_1_message_objects()
    demo_format_2_tuples()
    demo_format_3_dict()
    demo_format_4_single_string()

    print("\n--- 2. 多轮对话：为什么要传历史 ---")
    demo_multi_turn()

    print("\n--- 3. 对比总结 ---")
    demo_comparison_table()
