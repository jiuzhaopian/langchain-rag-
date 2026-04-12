"""
03_deepseek.py
DeepSeek 官方集成 - ChatDeepSeek

使用 langchain-deepseek 官方包，而不是 OpenAI 兼容模式。
这是 LangChain 官方推荐的 DeepSeek 集成方式。

参考文档：
  - 官方集成文档: https://docs.langchain.com/oss/python/integrations/chat/deepseek
  - API Key: https://platform.deepseek.com/api_keys

安装：
  pip install langchain-deepseek
"""

import os


def demo_deepseek():
    """
    使用 ChatDeepSeek（官方集成）
    参考: https://docs.langchain.com/oss/python/integrations/chat/deepseek
    """
    from langchain_deepseek import ChatDeepSeek

    # 初始化 - 支持两种模型
    llm = ChatDeepSeek(
        model="deepseek-chat",        # DeepSeek-V3（支持 tool calling、structured output）
        # model="deepseek-reasoner",  # DeepSeek-R1（推理模型，不支持 tool calling）
        temperature=0.7,
    )

    # invoke
    response = llm.invoke("用一句话介绍 LangChain")

    # 输出完整响应 JSON
    print(response.model_dump_json(indent=2))
    print(f"[ChatDeepSeek invoke] {response.content}")
    print(f"类型: {type(response).__name__}")
    print(f"模型: {response.response_metadata.get('model_name', 'N/A')}")
    print(f"Token: {response.usage_metadata}")


if __name__ == "__main__":
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("请设置 DEEPSEEK_API_KEY 环境变量")
        print("  export DEEPSEEK_API_KEY='sk-...'")
        print("\nAPI Key 获取: https://platform.deepseek.com/api_keys")
        exit(1)

    print("=== DeepSeek 官方集成 (ChatDeepSeek) ===\n")
    demo_deepseek()
