"""
01_basic_chain.py - LCEL 基本链

LCEL（LangChain Expression Language）用 | 管道符组合 Runnable，构成链。
所有 LangChain 组件（ChatModel、PromptTemplate、OutputParser 等）都是 Runnable，
可以用 | 串联，这就是现代 LangChain 中 Chain 的写法。

参考文档：
  - RunnableSequence: https://reference.langchain.com/python/langchain-core/runnables/base/RunnableSequence
  - RunnableLambda: https://reference.langchain.com/python/langchain-core/runnables/base/RunnableLambda
  - StrOutputParser: https://reference.langchain.com/python/langchain-core/output_parsers/string/StrOutputParser
  - ChatPromptTemplate: https://reference.langchain.com/python/langchain-core/prompts/chat/ChatPromptTemplate
  - 源码: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/runnables/

安装：
  pip install langchain-core langchain-community zhipuai
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


# ========================= 模型配置 =========================

def get_llm():
    from langchain_community.chat_models import ChatZhipuAI
    import os
    return ChatZhipuAI(
        model="glm-4.7",
        api_key=os.environ.get("ZHIPUAI_API_KEY"),
    )


# ============================================================
# 演示 1：基本三段式链
# ============================================================

def demo_basic_chain():
    """
    prompt | llm | parser 是最常用的三段式链。
    | 是 Python 的位或运算符，被 LangChain 的 Runnable 重载为管道操作：
    左边的输出自动作为右边的输入。

    每一环的输入输出类型：
      prompt:    dict[str, Any] -> PromptValue
      llm:       PromptValue      -> AIMessage
      parser:    AIMessage        -> str / dict / Pydantic

    注意：| 右侧可以是任何 callable（函数），LCEL 会自动包装为 RunnableLambda。
    """
    llm = get_llm()

    # 两段式：prompt | llm
    prompt = ChatPromptTemplate.from_messages([
        ("human", "用一句话解释什么是{concept}"),
    ])
    chain1 = prompt | llm
    result1 = chain1.invoke({"concept": "机器学习"})
    print("=== 两段式：prompt | llm ===")
    print(f"chain 类型: {type(chain1).__name__}")  # RunnableSequence
    print(f"结果类型: {type(result1).__name__}")    # AIMessage
    print(f"结果: {result1.content}")
    print()

    # 三段式：prompt | llm | parser
    chain2 = prompt | llm | StrOutputParser()
    result2 = chain2.invoke({"concept": "机器学习"})
    print("=== 三段式：prompt | llm | parser ===")
    print(f"结果类型: {type(result2).__name__}")    # str
    print(f"结果: {result2}")
    print()


# ============================================================
# 演示 2：普通函数可以直接入链
# ============================================================

def demo_function_in_chain():
    """
    | 右侧可以是任何 callable（函数），LCEL 源码中的 coerce_to_runnable()
    会自动把 callable 包装为 RunnableLambda，不需要手动包装。

    callable 放在 | 的左侧也可以，因为 Runnable 实现了 __ror__ 方法，
    当 Runnable 在 | 右侧时 __ror__ 被调用，同样会 coerce_to_runnable() 自动包装。


    所以以下三种写法等价：
      prompt | RunnableLambda(my_func) | llm    # 显式包装
      prompt | my_func | llm                   # 自动包装，推荐
    """
    print("=== 普通函数入链 ===")
    llm = get_llm()
    # TODO
    pass


# ============================================================
# 演示 3：chain 自动支持 batch 和 stream
# ============================================================

def demo_batch_stream():
    """
    用 | 组成的 chain 自动支持 batch() 和 stream()，不需要额外代码。
    这是 LCEL 的核心优势之一。
    """
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("human", "{topic}是什么？一句话回答"),
    ])

    chain = prompt | llm | StrOutputParser()

    # batch: 批量处理
    print("=== batch ===")
    results = chain.batch([{"topic": "Python"}, {"topic": "Java"}])
    for i, r in enumerate(results):
        print(f"  [{i}] {r}")
    print()

    # stream: 流式输出
    print("=== stream ===")
    for chunk in chain.stream({"topic": "AI"}):
        print(chunk, end="", flush=True)
    print("\n")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    demo_basic_chain()
    demo_function_in_chain()
    demo_batch_stream()
