"""
07_stream.py
流式调用 - 逐 token 输出，提升用户体验

Chat Models 提供两种流式方式：
  - stream(): 同步流式，返回迭代器
  - astream(): 异步流式，返回异步迭代器（适合 FastAPI 等异步框架）

适用场景：对话界面、长时间生成的文本、需要实时反馈的场景

参考文档：
  - Models stream: https://docs.langchain.com/oss/python/langchain/models#stream
"""

import os


def demo_stream():
    """
    同步流式调用 - stream()
    每个 chunk 是 AIMessageChunk 对象，有 .content 属性
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        temperature=0.7,
    )

    print("=== 同步流式 stream() ===")
    for chunk in llm.stream("写一首关于春天的五言绝句"):
        print(chunk.content, end="", flush=True)
    print("\n")


def demo_stream_ollama():
    """
    Ollama 流式调用（无需 API Key）
    """
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model="qwen3:8b", temperature=0.7)

    print("=== Ollama 流式 stream() ===")
    for chunk in llm.stream("写一首关于春天的五言绝句"):
        print(chunk.content, end="", flush=True)
    print("\n")


async def demo_astream():
    """
    异步流式调用 - astream()
    适合 FastAPI、asyncio 场景
    """
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model="qwen3:8b", temperature=0.7)

    print("=== 异步流式 astream() ===")
    async for chunk in llm.astream("写一首关于夏天的五言绝句"):
        print(chunk.content, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    # DashScope 流式
    print("--- 1. DashScope 流式 ---")
    demo_stream()

    # Ollama 流式（无需 API Key）
    print("--- 2. Ollama 流式 ---")
    demo_stream_ollama()

    # 异步流式
    print("--- 3. Ollama 异步流式 ---")
    import asyncio
    asyncio.run(demo_astream())

