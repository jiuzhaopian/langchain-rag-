"""
LangChain Output Parsers - String & JSON (基础解析器)

Output Parser 用于将 LLM 的原始输出转换为结构化数据。
最简单的两个：StrOutputParser 和 JsonOutputParser。

核心概念：
  - StrOutputParser: 提取纯文本字符串（去掉 message 包装）
  - JsonOutputParser: 将文本解析为 JSON dict
  - 所有 OutputParser 都是 Runnable，可用 | 串联

参考文档：
  - StrOutputParser API: https://reference.langchain.com/python/langchain-core/output_parsers/string/StrOutputParser
  - JsonOutputParser API: https://reference.langchain.com/python/langchain-core/output_parsers/json/JsonOutputParser
  - 源码: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/output_parsers/

安装：
  pip install langchain-core langchain-community zhipuai
"""

import json
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate


# ========================= 模型配置 =========================

def get_llm():
    """延迟加载，避免无 key 环境报错"""
    from langchain_community.chat_models import ChatTongyi
    import os
    llm = ChatTongyi(
        model="qwen-plus",
        dashscope_api_key=os.environ.get("ali_API_KEY"),
    )
    return llm


# ============================================================
# 演示 1：StrOutputParser (提取纯文本)
# ============================================================

def demo_str_parser():
    """
    StrOutputParser 是最简单的解析器：
    - 输入：AIMessage（或字符串）
    - 输出：纯文本字符串
    - 作用：去掉 message 包装，只保留 .content

    在 LCEL 链式调用中，parser 放在 llm 之后：
      prompt | llm | parser
    这是 LangChain 表达式语言的核心范式。
    """
    llm = get_llm()
    parser = StrOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("human", "用一句话介绍 Python"),
    ])

    # 链式调用：prompt -> llm -> parser
    chain = prompt | llm | parser
    result = chain.invoke({})

    print("=== StrOutputParser ===")
    print(f"结果类型: {type(result)}")
    print(f"结果内容: {result}")
    print()

    # 对比：不用 parser 时返回的是 AIMessage
    raw = (prompt | llm).invoke({})
    print(f"不用 parser: 返回 {type(raw).__name__}，需要用 .content 取值")
    print(f"不用 parser .content: {raw.content}")
    print()


# ============================================================
# 演示 2：JsonOutputParser (解析 JSON)
# ============================================================

def demo_json_parser():
    """
    JsonOutputParser 让 LLM 返回 JSON 格式，并解析为 dict。
    - 自动在 prompt 中注入 "返回 JSON" 的格式指令
    - invoke 返回 Python dict（注意：不做类型验证）
    - 依赖 LLM 遵循格式指令，不保证 100% 成功

    需要严格类型验证时，用 PydanticOutputParser（见 02_pydantic_parser.py）。
    """
    llm = get_llm()
    parser = JsonOutputParser()

    # 查看自动注入的格式指令
    print("=== JsonOutputParser 格式指令 ===")
    print(parser.get_format_instructions())
    print()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个信息提取助手。{format_instructions}"),
        ("human", "提取以下文本中的信息：张三，28岁，北京，软件工程师"),
    ])

    chain = prompt | llm | parser
    result = chain.invoke({
        "format_instructions": parser.get_format_instructions(),
    })

    print("=== JsonOutputParser 结果 ===")
    print(f"结果类型: {type(result)}")
    print(f"结果内容: {json.dumps(result, ensure_ascii=False, indent=2)}")
    print()


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    demo_str_parser()
    demo_json_parser()