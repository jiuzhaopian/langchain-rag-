"""03_list_parser.py - List Parsers (列表解析器)

LangChain 提供多种列表解析器，将 LLM 输出解析为 Python list。
注意：ListOutputParser 是抽象基类，不能直接实例化。
实际使用的是它的子类：
  - CommaSeparatedListOutputParser: 逗号分隔列表
  - NumberedListOutputParser: 编号列表（1. xxx）
  - MarkdownListOutputParser: Markdown 列表（- xxx）

适合场景：让 LLM 列出一组事物（城市、框架、步骤等）。

参考文档：
  - CommaSeparatedListOutputParser: https://reference.langchain.com/python/langchain-core/output_parsers/list/CommaSeparatedListOutputParser
  - 源码: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/output_parsers/list.py

安装：
  pip install langchain-core langchain-community zhipuai
"""

from langchain_core.output_parsers import (
    CommaSeparatedListOutputParser,
    StrOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate


# ========================= 模型配置 =========================

def get_llm():
    from langchain_community.chat_models import ChatZhipuAI
    import os
    return ChatZhipuAI(
        model="glm-4.7",
        api_key=os.environ.get("ZHIPUAI_API_KEY"),
    )


# ============================================================
# 演示 1：CommaSeparatedListOutputParser
# ============================================================

def demo_comma_separated():
    """
    CommaSeparatedListOutputParser：
    - 自动注入格式指令："返回逗号分隔的列表"
    - 将 LLM 输出按逗号拆分为 list[str]
    - 适合简单的枚举场景
    """
    llm = get_llm()
    parser = CommaSeparatedListOutputParser()

    print("=== CommaSeparatedListOutputParser 格式指令 ===")
    print(parser.get_format_instructions())
    print()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个编程助手。{format_instructions}"),
        ("human", "列出 5 个 Python Web 框架"),
    ])

    chain = prompt | llm | parser
    result = chain.invoke({
        "format_instructions": parser.get_format_instructions(),
    })

    print("=== 结果 ===")
    print(f"类型: {type(result).__name__}")
    print(f"列表: {result}")
    print(f"元素数: {len(result)}")
    print()


# ============================================================
# 演示 2：与 StrOutputParser 对比
# ============================================================

def demo_compare():
    """
    对比两种 parser 对同一输出的处理：
    - StrOutputParser: 返回原始字符串
    - CommaSeparatedListOutputParser: 按逗号拆分为 list
    """
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("human", "列出 3 种前端框架，用逗号分隔"),
    ])

    # 方式 1：StrOutputParser 返回原始字符串
    text = (prompt | llm | StrOutputParser()).invoke({})
    print("=== StrOutputParser ===")
    print(f"类型: {type(text).__name__}")
    print(f"内容: '{text}'")
    print()

    # 方式 2：CommaSeparatedListOutputParser 拆分为 list
    lst = (prompt | llm | CommaSeparatedListOutputParser()).invoke({
        "format_instructions": CommaSeparatedListOutputParser().get_format_instructions(),
    })
    print("=== CommaSeparatedListOutputParser ===")
    print(f"类型: {type(lst).__name__}")
    print(f"列表: {lst}")
    print()


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    demo_comma_separated()
    demo_compare()
