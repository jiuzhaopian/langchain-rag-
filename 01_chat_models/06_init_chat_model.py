"""
06_init_chat_model.py
init_chat_model() - LangChain 1.x 推荐的通用模型初始化器

init_chat_model 的优势：
  - 统一接口：一个函数初始化所有 provider 的模型
  - 自动推断 provider：通过模型名前缀自动选择（如 "gpt-" → OpenAI）
  - 显式指定 provider：model_provider 参数
  - 可配置字段：configurable_fields 支持运行时切换模型

参考文档：
  - Models 概览: https://docs.langchain.com/oss/python/langchain/models

安装：
  pip install langchain
"""

import os


def demo_basic():
    """
    基本用法 - 初始化模型
    """
    from langchain.chat_models import init_chat_model

    # 方式 1：自动推断 provider（模型名前缀判断）
    # "gpt-" → OpenAI, "claude-" → Anthropic, "gemini-" → Google
    # model = init_chat_model("gpt-4o-mini")

    # 方式 2：显式指定 provider（推荐，避免歧义）
    pass


def demo_providers():
    """
    不同 provider 的使用方式
    """
    from langchain.chat_models import init_chat_model
    model = init_chat_model(
        "qwen-plus",
        model_provider="tongyi",  # provider 名称,但支持的provider是在底层代码中写好的
        temperature=0.7,
    )
    response = model.invoke("用一句话介绍 LangChain")
    print(f"[init_chat_model + tongyi] {response.content}")


# def demo_configurable():
#     """
#     运行时切换模型 - configurable fields
#     """
#     from langchain.chat_models import init_chat_model
#
#     # 创建可配置模型
#     configurable = init_chat_model(
#         temperature=0,
#         configurable_fields=("model", "model_provider"),
#     )
#
#     # 调用时指定不同模型,另外 invoke时的config参数用法了解
#     response1 = configurable.invoke(
#         "用 3 个字形容猫",
#         config={"configurable": {"model": "qwen3:8b", "model_provider": "ollama"}},
#     )
#     print(f"[动态 - ollama qwen3:8b] {response1.content}")
#

if __name__ == "__main__":
    print("=== init_chat_model 通用初始化器 ===\n")

    # 基本用法
    print("--- 1. 基本用法 ---")
    #demo_basic()
    print()


    # Provider 列表
    print("--- 2. ollama Provider  ---")
    demo_providers()
    print()

    # 可配置模型
    print("--- 3. 运行时切换模型 ---")
    demo_configurable()

