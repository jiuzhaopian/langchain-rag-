"""
03_passthrough.py - RunnablePassthrough 数据透传

RunnablePassthrough 用于在链中"透传"数据：原样传递输入，或保留原字段的同时添加新字段。
最典型的场景是 RAG（检索增强生成）。

执行流程图解（RAG 链）：
  rag_chain = {
      "context": retriever,       # 自定义函数，自动包装为 RunnableLambda
      "question": RunnablePassthrough(),  # 原样传递
  } | prompt | llm | StrOutputParser()

  调用 rag_chain.invoke("什么是 LangChain") 的执行过程：
    1. 输入 "什么是 LangChain" 是一个字符串
    2. Python 计算 dict | prompt 时，dict 会被自动转为 RunnableParallel
       注意：dict 在 | 的左侧或右侧都会被自动转换，因为 Runnable 同时
       实现了 __or__（Runnable 在左侧，coerce 右侧）和 __ror__（Runnable 在右侧，coerce 左侧）
    3. RunnableParallel 将输入分发给两个分支：
       - "context" 分支：retriever("什么是 LangChain") 返回检索到的文档
       - "question" 分支：RunnablePassthrough() 原样返回 "什么是 LangChain"
    4. 两个分支结果合并为 dict：{"context": "...", "question": "什么是 LangChain"}
    5. 这个 dict 作为输入传给 prompt（ChatPromptTemplate 用这些变量填充模板）
    6. prompt 输出 -> llm -> StrOutputParser
    7. 最终返回解析后的字符串

参考文档：
  - RunnablePassthrough: https://reference.langchain.com/python/langchain-core/runnables/passthrough/RunnablePassthrough
  - 源码: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/runnables/

安装：
  pip install langchain-core langchain-community zhipuai
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# ========================= 模型配置 =========================

def get_llm():
    from langchain_community.chat_models import ChatZhipuAI
    import os
    return ChatZhipuAI(
        model="glm-4.7",
        api_key=os.environ.get("ZHIPUAI_API_KEY"),
    )


# ============================================================
# 演示 1：RunnablePassthrough 基本行为
# ============================================================

def demo_passthrough_basic():
    """
    RunnablePassthrough() 就是 identity function（恒等函数）：
    输入什么就输出什么。
    """
    passthrough = RunnablePassthrough()

    r1 = passthrough.invoke("hello")
    print("=== RunnablePassthrough 基本行为 ===")
    print(f"passthrough.invoke('hello') = '{r1}'")

    r2 = passthrough.invoke({"name": "张三", "age": 25})
    print(f"passthrough.invoke(dict) = {r2}")
    print()


# ============================================================
# 演示 2：RunnablePassthrough.assign
# ============================================================

def demo_assign():
    """
    RunnablePassthrough.assign(k=fn) 保留原始输入的所有字段，
    同时添加新字段 k（值为 fn 处理后的结果）。

    """
    # TODO
    pass


def demo_rag_flow():
    """
    RAG 链的完整流程。

    关键：为什么这里需要 RunnablePassthrough？
    因为 | 管道中的 dict 会被转为 RunnableParallel，
    dict 的每个 value 会接收整个输入作为参数。
    - retriever("什么是 LangChain") -> 返回文档字符串（正确）
    - 但如果直接写 "question": lambda x: x，输入是字符串时没问题
      输入是 dict 时就会把整个 dict 作为 question 的值（可能不是你想要的）

    RunnablePassthrough() 确保无论输入是什么，都原样透传。
    """
    llm = get_llm()
    # TODO
    pass


# 测试 passthrough 透传的是什么
def test_passthrough():
    llm = get_llm()

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "根据上下文回答问题。\n\n上下文：{context}"),
        ("human", "{question}"),
    ])

    parser = StrOutputParser()

    chain = {
        "context": lambda x:f"上下文是:{x}",
        "question": RunnablePassthrough(),
    } | prompt_template | llm | parser | {
        "print":lambda x:print(x),
        "result": RunnablePassthrough(),
    }

    result = chain.invoke("讲一个10字以内的笑话")
    print(f"执行结果为: {result}")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    #demo_passthrough_basic()
    #demo_assign()
    #demo_rag_flow()
    test_passthrough()
