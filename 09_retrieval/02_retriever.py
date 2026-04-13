"""
02_retriever.py - Retriever 检索器体系

Retriever 是 LangChain 中统一的检索接口，将查询（query）转换为相关文档列表。
所有 Retriever 都实现 invoke(query) → list[Document] 的统一接口。

本 Demo 涵盖：
  - VectorStoreRetriever  基于向量相似度的检索（dense retrieval）
  - BM25Retriever         基于关键词匹配的检索（sparse retrieval）
  - MultiQueryRetriever   用 LLM 生成多个查询变体，提高召回率

参考文档：
  - Retriever 接口: https://python.langchain.com/api_reference/core/retrievers.html
  - BM25Retriever: https://python.langchain.com/api_reference/community/retrievers/langchain_community.retrievers.bm25.BM25Retriever.html
  - MultiQueryRetriever: https://python.langchain.com/api_reference/langchain/retrievers/langchain.retrievers.multi_query.MultiQueryRetriever.html

安装：
  pip install langchain-core langchain-community rank_bm25
"""

import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore


# ============================================================
# 辅助
# ============================================================

class FakeEmbedding:
    """伪 Embedding，仅用于 API 演示"""
    DIMENSION = 128

    def embed_documents(self, texts):
        return [
            np.array([hash(t + str(i)) % 10000 / 10000.0 for i in range(self.DIMENSION)])
            for t in texts
        ]

    def embed_query(self, text):
        return np.array(
            [hash(text + str(i)) % 10000 / 10000.0 for i in range(self.DIMENSION)]
        )


SAMPLE_DOCUMENTS = [
    Document(page_content="LangChain 是一个用于构建 LLM 应用的 Python 框架", metadata={"topic": "framework"}),
    Document(page_content="Chroma 是一个开源的向量数据库，支持本地部署", metadata={"topic": "database"}),
    Document(page_content="FAISS 是 Facebook 开发的高效向量相似度搜索库", metadata={"topic": "database"}),
    Document(page_content="BM25 是一种经典的关键词匹配检索算法", metadata={"topic": "algorithm"}),
    Document(page_content="RAG 结合检索和生成，让 LLM 基于外部知识回答问题", metadata={"topic": "application"}),
    Document(page_content="Embedding 模型将文本转换为高维向量表示", metadata={"topic": "embedding"}),
    Document(page_content="Python requests 库用于发送 HTTP 请求", metadata={"topic": "python"}),
]


# ============================================================
# Demo 函数
# ============================================================

def demo_vector_store_retriever():
    """Demo 1: VectorStoreRetriever — 基于向量相似度的检索"""
    print("\n" + "=" * 60)
    print("Demo 1: VectorStoreRetriever（向量相似度检索）")
    print("=" * 60)

    embedding = FakeEmbedding()
    vs = InMemoryVectorStore.from_documents(SAMPLE_DOCUMENTS, embedding)

    # 通过 as_retriever() 将 VectorStore 转换为 Retriever
    # k 控制返回的文档数量，默认 k=4
    retriever = vs.as_retriever(search_kwargs={"k": 3})

    query = "向量数据库推荐"
    results = retriever.invoke(query)
    print(f"查询: '{query}'，返回 {len(results)} 条结果:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. [{doc.metadata['topic']}] {doc.page_content}")

    # 也可以用 search_type 指定搜索策略
    # "similarity"（默认）: 按相似度排序
    # "mmr": 最大边际相关性，兼顾相关性和多样性
    retriever_mmr = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10}
    )
    print(f"\nMMR 检索（兼顾多样性）:")
    results_mmr = retriever_mmr.invoke(query)
    for i, doc in enumerate(results_mmr, 1):
        print(f"  {i}. [{doc.metadata['topic']}] {doc.page_content}")


def demo_bm25_retriever():
    """Demo 2: BM25Retriever — 基于关键词匹配的检索"""
    print("\n" + "=" * 60)
    print("Demo 2: BM25Retriever（关键词匹配检索）")
    print("=" * 60)

    try:
        from langchain_community.retrievers import BM25Retriever
    except ImportError:
        print("⚠️  BM25Retriever 需要 rank_bm25: pip install rank_bm25")
        return

    # Move the BM25Retriever usage inside the try to catch import AND runtime issues
    try:
        retriever = BM25Retriever.from_documents(SAMPLE_DOCUMENTS, k=3)
    except ImportError:
        print("⚠️  BM25Retriever 在当前环境中不可用，需要安装 rank_bm25")
        return

    # BM25 基于词频统计，不需要 Embedding
    # 适合关键词明确的场景，如技术文档搜索
    retriever = BM25Retriever.from_documents(SAMPLE_DOCUMENTS, k=3)

    query = "向量数据库"
    results = retriever.invoke(query)
    print(f"查询: '{query}'，返回 {len(results)} 条结果:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. [{doc.metadata['topic']}] {doc.page_content}")

    # BM25 对模糊查询不如向量检索
    print(f"\n模糊查询: '有什么推荐的数据库技术'")
    results = retriever.invoke("有什么推荐的数据库技术")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. [{doc.metadata['topic']}] {doc.page_content}")
    print("→ BM25 依赖关键词匹配，模糊查询效果不如向量检索")


def demo_multi_query_retriever():
    """Demo 3: MultiQueryRetriever — 用 LLM 生成多个查询变体"""
    print("\n" + "=" * 60)
    print("Demo 3: MultiQueryRetriever（多查询变体检索）")
    print("=" * 60)

    # MultiQueryRetriever 需要一个 LLM 来生成查询变体
    # 它会对原始查询生成 3-4 个改写版本，然后分别检索并合并去重
    # 这样可以提高召回率，特别是当用户查询表述不够准确时
    #
    # 原理：
    #   用户查询 "向量数据库" → LLM 生成:
    #     1. "vector database"
    #     2. "向量存储和检索"
    #     3. "embedding 数据库"
    #   → 对每个变体分别检索 → 合并去重 → 返回结果

    try:
        from langchain.retrievers.multi_query import MultiQueryRetriever
    except ImportError:
        # langchain-core 0.4.x 中 MultiQueryRetriever 可能在不同包中
        try:
            from langchain_community.retrievers.multi_query import MultiQueryRetriever
        except ImportError:
            print("⚠️  MultiQueryRetriever 在当前版本中不可用")
            print("   需要: pip install langchain")
            print("\n用法示例:")
            print("""
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from langchain_community.chat_models import ChatZhipuAI

    llm = ChatZhipuAI(model="glm-4", api_key="your_key")
    retriever = BM25Retriever.from_documents(documents, k=3)
    multi_query = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
    results = multi_query.invoke("向量数据库")
            """)
            return

    try:
        from langchain_community.chat_models import ChatZhipuAI

        api_key = __import__("os").getenv("ZHIPUAI_API_KEY")
        if not api_key:
            print("⚠️  未设置 ZHIPUAI_API_KEY 环境变量，跳过 MultiQueryRetriever 演示")
            return

        llm = ChatZhipuAI(model="glm-4", api_key=api_key)
        base_retriever = BM25Retriever.from_documents(SAMPLE_DOCUMENTS, k=2)
        multi_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever, llm=llm
        )

        query = "向量数据库"
        print(f"原始查询: '{query}'")
        print("LLM 生成查询变体中...（可能需要几秒）")
        results = multi_retriever.invoke(query)
        print(f"合并后返回 {len(results)} 条结果:")
        for i, doc in enumerate(results, 1):
            print(f"  {i}. [{doc.metadata['topic']}] {doc.page_content}")
    except Exception as e:
        print(f"⚠️  演示跳过: {e}")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    demo_vector_store_retriever()
    demo_bm25_retriever()
    demo_multi_query_retriever()
    print("\n✅ 所有 Demo 运行完成")
