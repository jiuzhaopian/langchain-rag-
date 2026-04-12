"""
01_openai_compatible.py
ChatOpenAI + base_url 兼容 OpenAI 协议的服务

适用场景：任何兼容 OpenAI Chat Completions API 的服务
  - 通义千问 DashScope（也支持原生接口，见 02_dashscope_native.py）
  - DeepSeek
  - 硅基流动 SiliconFlow
  - vLLM、LiteLLM 等自部署服务

原理：OpenAI 定义了 Chat Completions API 标准，很多国产模型服务兼容这个标准。
      ChatOpenAI 的 base_url 参数可以指向这些服务的端点。

参考文档：
  - ChatOpenAI: https://python.langchain.com/docs/integrations/chat/openai/
  - 通用 Models: https://docs.langchain.com/oss/python/langchain/models

安装：
  pip install langchain
  pip install langchain-openai
"""

import os


# ============================================================
# 1. 通过 DashScope 的 OpenAI 兼容接口调用通义千问
# ============================================================

def demo_dashscope_openai_compatible():
    """
    通义千问的 DashScope 提供了 OpenAI 兼容接口
    端点: https://dashscope.aliyuncs.com/compatible-mode/v1
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="qwen-plus",                              # 模型名
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/",  # DashScope 兼容端点
        api_key=os.environ.get("DASHSCOPE_API_KEY"),     # 从环境变量读取
        temperature=0.7,
    )

    # invoke - 非流式调用
    response = llm.invoke("用一句话介绍 LangChain")
    print(response.model_dump_json(indent=2))
    print(f"[DashScope 兼容模式] {response.content}")
    # response 是 AIMessage 对象
    print(f"类型: {type(response).__name__}")
    print(f"Token: {response.usage_metadata}")


# ============================================================
# 2. 通过 DeepSeek 的 OpenAI 兼容接口调用
# ============================================================

def demo_deepseek():
    """
    DeepSeek 的 API 完全兼容 OpenAI Chat Completions 协议
    端点: https://api.deepseek.com（官方文档推荐此地址，无需手动拼 /v1）
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="deepseek-chat",                          # DeepSeek 聊天模型
        base_url="https://api.deepseek.com",             # DeepSeek 端点
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        temperature=0.7,
    )

    response = llm.invoke("用一句话介绍 LangChain")
    print(response.model_dump_json(indent=2))
    print(f"[DeepSeek] {response.content}")
    print(f"类型: {type(response).__name__}")
    print(f"Token: {response.usage_metadata}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    # DashScope 兼容模式
    if os.environ.get("DASHSCOPE_API_KEY"):
        print("=== 1. DashScope OpenAI 兼容模式 ===")
        demo_dashscope_openai_compatible()
    else:
        print("跳过 DashScope：未设置 DASHSCOPE_API_KEY")

    print()

    # DeepSeek
    if os.environ.get("DEEPSEEK_API_KEY"):
        print("=== 2. DeepSeek OpenAI 兼容模式===")
        demo_deepseek()
    else:
        print("跳过 DeepSeek：未设置 DEEPSEEK_API_KEY")
