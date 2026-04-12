"""
08_batch.py
批量调用 - 同时处理多个请求

Chat Models 提供两种批量调用方式：
  - batch(): 同步批量，传入消息列表，返回响应列表
  - abatch(): 异步批量，适合 FastAPI 等异步框架

异步批量（abatch）和同步批量（batch）的核心区别在于并发实现方式：
    1.同步批量（batch）- 线程池并发：
        # 对外：同步函数，阻塞直到所有结果返回
        # 内部：使用线程池实现并发请求
        # 适合：同步脚本、Jupyter notebook

    2.异步批量（abatch）- 协程并发：
        # 对外：异步函数，需在 async 函数中 await
        # 内部：使用 asyncio 实现并发
        # 适合：FastAPI、异步Web框架、异步脚本

    3.重要：
        # - batch() 和 abatch() 都是并发！
        # - 普通 for 循环里的 invoke() 才是串行
        # - abatch() 是原生异步，比线程池更高效（无线程切换开销）
        # - Ollama 默认 OLLAMA_NUM_PARALLEL=1，会强制串行
        #   需设置环境变量才能看到并发加速效果
        #   设置方式：
        #     macOS / Linux:
        #       OLLAMA_NUM_PARALLEL=4 ollama serve
        #     或者先 export:
        #       export OLLAMA_NUM_PARALLEL=4
        #       ollama serve
        #     Windows PowerShell:
        #       $env:OLLAMA_NUM_PARALLEL="4"; ollama serve
        #     Windows CMD:
        #       set OLLAMA_NUM_PARALLEL=4 && ollama serve
        #     Linux systemd:
        #       编辑 /etc/systemd/system/ollama.service
        #       在 [Service] 下添加 Environment="OLLAMA_NUM_PARALLEL=4"

    4.可视化对比（假设服务端支持并发）：
        普通 for 循环串行：
        [请求1: ████████ 2秒] [请求2: ████████ 2秒] [请求3: ████████ 2秒]
        总耗时：6秒（串行）

        batch() / abatch() 并发：
        [请求1: ████████ 2秒]
        [请求2: ████████ 2秒] ← 同时开始
        [请求3: ████████ 2秒] ← 同时开始
        总耗时：2秒（并发）

适用场景：同时处理多个查询、批量翻译、批量摘要等

参考文档：
  - Models 概览: https://docs.langchain.com/oss/python/langchain/models
"""

import os


def demo_batch_ollama():
    """
    Ollama 同步批量调用 - batch()
    输入: List[str] 或 List[List[Message]]
    输出: List[AIMessage]，与输入一一对应
    """
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model="qwen3:8b", temperature=0)

    print("=== Ollama 同步批量 batch() ===")
    responses = llm.batch([
        "1+1=? 只回答数字",
        "2+2=? 只回答数字",
        "3+3=? 只回答数字",
    ])
    for i, resp in enumerate(responses):
        print(f"  请求 {i + 1}: {resp.content}")


def demo_abatch_ollama():
    """
    Ollama 异步批量调用 - abatch()
    与 batch() 一样用 Ollama，但通过 asyncio 并发执行
    适合 FastAPI、asyncio 等异步场景
    """
    from langchain_ollama import ChatOllama
    import asyncio

    llm = ChatOllama(model="qwen3:8b", temperature=0)

    print("=== Ollama 异步批量 abatch() ===")

    async def run_async():
        responses = await llm.abatch([
            "用 3 个字形容猫",
            "用 3 个字形容狗",
            "用 3 个字形容鱼",
        ])
        for i, resp in enumerate(responses):
            print(f"  请求 {i + 1}: {resp.content}")

    asyncio.run(run_async())


if __name__ == "__main__":
    # Ollama 同步批量（无需 API Key）
    print("--- 1. Ollama 同步批量 ---")
    demo_batch_ollama()
    print()

    # Ollama 异步批量
    print("--- 2. Ollama 异步批量 (abatch) ---")
    demo_abatch_ollama()

