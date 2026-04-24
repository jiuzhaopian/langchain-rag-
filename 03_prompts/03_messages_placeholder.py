"""
03_messages_placeholder.py
LangChain Prompts - MessagesPlaceholder (消息占位符)

MessagesPlaceholder 用于在 ChatPromptTemplate 中插入一组消息。
最常见的场景是多轮对话：把对话历史作为变量传入。

为什么需要它？
  - 普通变量 {"role": "human", "content": "{history}"} 只能插入一条消息
  - MessagesPlaceholder 可以插入任意数量的消息（多条 human/ai/system 交替）
  - 保持消息的类型信息（human/ai），而不是拼成纯文本丢失结构

参考文档：
  - MessagesPlaceholder API: https://reference.langchain.com/python/langchain-core/prompts/chat/MessagesPlaceholder
  - 源码: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/prompts/chat.py

安装：
  pip install langchain-core zhipuai
"""

import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


# ============================================================
# 公共：模型实例
# ============================================================

def get_llm(temperature=0):
    """获取智谱 GLM 模型实例"""
    from langchain_community.chat_models import ChatZhipuAI
    return ChatZhipuAI(model="glm-4.7", temperature=temperature)


# ============================================================
# 演示 1：基本用法 - 插入对话历史
# ============================================================

def demo_basic():
    """
    MessagesPlaceholder 最常见的用法：在模板中预留一个位置，
    运行时传入一组历史消息。
    """
    pass


# ============================================================
# 演示 2：与 LLM 链式调用（多轮对话）
# ============================================================

def demo_chain():
    """
    结合 LCEL 链式调用，实现多轮对话。
    """
    llm = get_llm()

    pass


# ============================================================
# 演示 3：optional 可选占位符
# ============================================================

def demo_optional():
    """
    MessagesPlaceholder(optional=True) 允许不传该变量。
    适用于：有些场景有历史，有些没有。
    """
    # optional=True：不传 history 也不会报错
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个翻译助手。"),
        MessagesPlaceholder("history", optional=True),
        ("human", "{input}"),
    ])

    # 不传 history
    result1 = prompt.invoke({"input": "Hello"})
    print("=== optional 可选占位符 ===")
    print(f"不传 history 时的消息数: {len(result1.to_messages())}")
    for msg in result1.to_messages():
        print(f"  [{msg.type}] {msg.content}")
    print()

    # 传 history
    result2 = prompt.invoke({
        "history": [
            HumanMessage(content="Hello"),
            AIMessage(content="你好！"),
        ],
        "input": "再见",
    })
    print(f"传入 history 时的消息数: {len(result2.to_messages())}")
    for msg in result2.to_messages():
        print(f"  [{msg.type}] {msg.content}")
    print()


# ============================================================
# 演示 4：MessagesPlaceholder vs 纯文本变量
# ============================================================

def demo_comparison():
    """
    对比两种方式处理对话历史：

    方式 A：纯文本变量 → 丢失消息类型信息
    方式 B：MessagesPlaceholder → 保留完整消息结构
    """
    # 方式 A：纯文本（不推荐）
    prompt_text = ChatPromptTemplate.from_messages([
        ("system", "你是助手。"),
        ("human", "历史对话：\n{history}\n\n当前问题：{input}"),
    ])

    # 方式 B：MessagesPlaceholder（推荐）
    prompt_ph = ChatPromptTemplate.from_messages([
        ("system", "你是助手。"),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])

    history = [
        HumanMessage(content="1+1等于几？"),
        AIMessage(content="等于2。"),
    ]

    result_a = prompt_text.invoke({
        "history": "用户: 1+1等于几？\nAI: 等于2。",
        "input": "我刚才问了什么？",
    })
    result_b = prompt_ph.invoke({
        "history": history,
        "input": "我刚才问了什么？",
    })

    print("=== MessagesPlaceholder vs 纯文本 ===")
    print(f"方式 A (纯文本) 的消息数: {len(result_a.to_messages())}")
    print(f"方式 B (Placeholder) 的消息数: {len(result_b.to_messages())}")
    print()
    print("方式 A 的问题：")
    print("  - 所有历史被拼成一条 human 消息，模型分不清哪句是用户说的、哪句是AI说的")
    print("  - 历史越长，prompt 越混乱")
    print()
    print("方式 B 的优势：")
    print("  - 每条历史消息保留原始类型 (human/ai)")
    print("  - 模型能准确区分用户输入和AI回复")
    print("  - 结构清晰，不受历史长度影响")
    print()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    if not os.environ.get("ZHIPUAI_API_KEY"):
        print("请设置 ZHIPUAI_API_KEY 环境变量")
        print("  export ZHIPUAI_API_KEY='...'")
        exit(1)

    demo_basic()
    demo_chain()
    demo_optional()
    demo_comparison()
