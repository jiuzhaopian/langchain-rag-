"""
02_chat_prompt_template.py
LangChain Prompts - ChatPromptTemplate (聊天模板)

ChatPromptTemplate 是 LangChain 中最常用的模板类，用于生成 Chat Model 所需的消息列表。
每个消息模板对应一种消息类型（system/human/ai），支持变量填充。

核心 API：
  - from_messages()    从消息列表创建（最常用）
  - from_template()    从单个字符串创建（快捷方式，默认为 human 消息）
  - invoke()           填充变量，返回 ChatPromptValue

参考文档：
  - ChatPromptTemplate API: https://reference.langchain.com/python/langchain-core/prompts/chat/ChatPromptTemplate
  - 源码: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/prompts/chat.py

安装：
  pip install langchain-core zhipuai
"""

import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# ============================================================
# 公共：模型实例（函数内延迟导入，避免无 key 环境报错）
# ============================================================

def get_llm(temperature=0):
    """获取智谱 GLM 模型实例"""
    from langchain_community.chat_models import ChatZhipuAI
    return ChatZhipuAI(model="glm-4.7", temperature=temperature)


# ============================================================
# 演示 1：from_messages 基本用法
# ============================================================

def demo_from_messages():
    """
    from_messages() 是最灵活的创建方式，支持多种消息格式：

    格式          示例                               说明
    ----          ----                               ----
    Message 对象  SystemMessage(content="...")        直接使用消息对象
    元组          ("system", "你是{role}")             (类型, 模板字符串)
    字典          {"role": "system", "content": "..."}  字典格式
    纯字符串      "你是{role}"                        默认为 human 消息
    """
    # TODO
    pass


# ============================================================
# 演示 2：from_template 快捷方式
# ============================================================

def demo_from_template():
    """
    from_template() 只能创建单条 human 消息的模板。
    如果只需要一条 human 消息，这个方式最简洁。
    """
    # TODO
    pass


# ============================================================
# 演示 3：ChatPromptValue 输出类型
# ============================================================

def demo_output_types():
    """
    ChatPromptTemplate.invoke() 返回 ChatPromptValue，
    对比 PromptTemplate 返回的 StringPromptValue：

    PromptTemplate     → StringPromptValue  → to_string() / to_messages()
    ChatPromptTemplate → ChatPromptValue    → to_string() / to_messages()

    区别：
      - StringPromptValue.to_messages() 返回 [HumanMessage]（纯文本包装）
      - ChatPromptValue.to_messages()   返回原始消息列表（保留类型信息）
    """
    # TODO
    pass


# ============================================================
# 演示 4：与 Chat Model 配合使用（LCEL 链式调用）
# ============================================================

def demo_with_llm():
    """
    PromptTemplate 和 Chat Model 都是 Runnable，
    可以用 | 管道符串联，实现 "模板填充 → 模型调用" 的一站式流程。

    这是 LangChain 的核心用法：LCEL (LangChain Expression Language)
    """
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}。"),
        ("human", "{question}"),
    ])

    # 用 | 串联成链：prompt → llm
    # TODO
    pass




# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    if not os.environ.get("ZHIPUAI_API_KEY"):
        print("请设置 ZHIPUAI_API_KEY 环境变量")
        print("  export ZHIPUAI_API_KEY='...'")
        exit(1)

    demo_from_messages()
    demo_from_template()
    demo_output_types()
    demo_with_llm()