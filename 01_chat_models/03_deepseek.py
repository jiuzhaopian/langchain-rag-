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

    pass


if __name__ == "__main__":
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("请设置 DEEPSEEK_API_KEY 环境变量")
        print("  export DEEPSEEK_API_KEY='sk-...'")
        print("\nAPI Key 获取: https://platform.deepseek.com/api_keys")
        exit(1)

    print("=== DeepSeek 官方集成 (ChatDeepSeek) ===\n")
    demo_deepseek()
