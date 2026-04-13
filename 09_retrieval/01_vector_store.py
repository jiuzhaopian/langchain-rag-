"""
01_vector_store.py - InMemoryVectorStore 向量存储基础

InMemoryVectorStore 是 LangChain 内置的内存向量数据库，适合开发测试和学习。
数据存储在内存中，程序退出后丢失。生产环境建议使用 Chroma、FAISS 等。

核心功能：
  - from_documents / from_texts  批量创建
  - add_documents / add_texts    追加文档
  - similarity_search            相似度搜索
  - similarity_search_with_score 带分数的相似度搜索
  - similarity_search_by_vector  按向量搜索
  - as_retriever()               转换为 Retriever
  - delete                       按 ID 删除文档

注意：InMemoryVectorStore 需要 Embedding 模型将文本转为向量。
本 demo 使用 FakeEmbedding（基于 hash）演示 API 用法，无需外部服务。
实际使用时替换为 Ollama bge-m3 或 OpenAI 等真实 Embedding。

参考文档：
  - InMemoryVectorStore: https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.in_memory.InMemoryVectorStore.html

安装：
  pip install langchain-core
"""

import uuid

import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore


# ============================================================
# 辅助：伪造 Embedding（仅用于演示，实际项目请用真实 Embedding）
# ============================================================

class FakeEmbedding:
    """
    基于 hash 的伪 Embedding，仅用于演示 VectorStore API。
    实际项目中请替换为真实的 Embedding 模型，例如：
      from langchain_community.embeddings import OllamaEmbeddings
      embedding = OllamaEmbeddings(model="bge-m3")
    """
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


def get_embedding():
    """获取 Embedding 实例（优先使用真实模型，不可用时降级为 FakeEmbedding）"""
    try:
        from langchain_community.embeddings import OllamaEmbeddings
        emb = OllamaEmbeddings(model="bge-m3")
        # 测试一下是否可用
        emb.embed_query("test")
        print("✅ 使用 Ollama bge-m3 Embedding")
        return emb
    except Exception:
        print("⚠️  Ollama 不可用，使用 FakeEmbedding（仅用于 API 演示）")
        return FakeEmbedding()


# ============================================================
# 示例文档
# ============================================================

SAMPLE_DOCUMENTS = [
    Document(page_content="LangChain 是一个用于构建 LLM 应用的 Python 框架", metadata={"source": "intro"}),
    Document(page_content="VectorStore 用于存储和检索文档的向量表示", metadata={"source": "retrieval"}),
    Document(page_content="Embedding 将文本转换为高维向量，语义相近的文本向量距离更近", metadata={"source": "embedding"}),
    Document(page_content="Retriever 从 VectorStore 中检索与查询最相关的文档", metadata={"source": "retrieval"}),
    Document(page_content="RAG（检索增强生成）结合检索和生成，让 LLM 基于外部知识回答问题", metadata={"source": "rag"}),
]


# ============================================================
# Demo 函数
# ============================================================

def demo_from_documents():
    """Demo 1: 从 Document 列表创建 VectorStore"""
    print("\n" + "=" * 60)
    print("Demo 1: InMemoryVectorStore.from_documents()")
    print("=" * 60)

    embedding = get_embedding()
    vs = InMemoryVectorStore.from_documents(SAMPLE_DOCUMENTS, embedding)

    print(f"已从 {len(SAMPLE_DOCUMENTS)} 个文档创建 VectorStore")
    # InMemoryVectorStore 内部存储了文档和向量
    print(f"内部文档数: {len(vs.store)}")


def demo_from_texts():
    """Demo 2: 从纯文本列表创建 VectorStore"""
    print("\n" + "=" * 60)
    print("Demo 2: InMemoryVectorStore.from_texts()")
    print("=" * 60)

    embedding = get_embedding()
    texts = ["Python 是一种编程语言", "Java 是一种编程语言", "猫是一种动物"]
    metadatas = [{"lang": "python"}, {"lang": "java"}, {"type": "animal"}]

    vs = InMemoryVectorStore.from_texts(texts, embedding, metadatas=metadatas)

    print(f"已从 {len(texts)} 段文本创建 VectorStore")
    # InMemoryVectorStore 内部用 dict 存储，每项包含 text、vector、metadata
    for key, entry in list(vs.store.items())[:3]:
        print(f"  - {entry['text'][:30]} | metadata: {entry.get('metadata', {})}")


def demo_add_documents():
    """Demo 3: 追加文档（add_documents / add_texts）"""
    print("\n" + "=" * 60)
    print("Demo 3: add_documents() / add_texts()")
    print("=" * 60)

    embedding = get_embedding()
    vs = InMemoryVectorStore.from_documents(
        [SAMPLE_DOCUMENTS[0]], embedding
    )
    print(f"初始文档数: {len(vs.store)}")

    # add_documents：追加 Document 对象
    new_ids = vs.add_documents(SAMPLE_DOCUMENTS[1:3])
    print(f"add_documents 追加了 {len(new_ids)} 个文档")
    print(f"返回的 ID: {new_ids}")

    # add_texts：追加纯文本
    new_ids = vs.add_texts(
        ["新增的文本内容 A", "新增的文本内容 B"],
        metadatas=[{"seq": 1}, {"seq": 2}]
    )
    print(f"add_texts 追加了 {len(new_ids)} 个文本")
    print(f"当前总文档数: {len(vs.store)}")


def demo_similarity_search():
    """Demo 4: 基础相似度搜索"""
    print("\n" + "=" * 60)
    print("Demo 4: similarity_search()")
    print("=" * 60)

    embedding = get_embedding()
    vs = InMemoryVectorStore.from_documents(SAMPLE_DOCUMENTS, embedding)

    # 默认 k=4
    results = vs.similarity_search("什么是 RAG", k=2)
    print(f"查询: '什么是 RAG'，返回 top 2:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. [{doc.metadata['source']}] {doc.page_content}")


def demo_similarity_search_with_score():
    """Demo 5: 带分数的相似度搜索"""
    print("\n" + "=" * 60)
    print("Demo 5: similarity_search_with_score()")
    print("=" * 60)

    embedding = get_embedding()
    vs = InMemoryVectorStore.from_documents(SAMPLE_DOCUMENTS, embedding)

    # 返回 (Document, score) 元组列表
    # InMemoryVectorStore 使用余弦相似度，分数范围 [0, 1]，1 表示完全相同
    results = vs.similarity_search_with_score("向量存储的用途", k=3)
    print(f"查询: '向量存储的用途'，返回 top 3（带分数）:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"  {i}. score={score:.4f} | [{doc.metadata['source']}] {doc.page_content[:40]}")


def demo_similarity_search_by_vector():
    """Demo 6: 按向量搜索（跳过文本 embedding 步骤）"""
    print("\n" + "=" * 60)
    print("Demo 6: similarity_search_by_vector()")
    print("=" * 60)

    embedding = get_embedding()
    vs = InMemoryVectorStore.from_documents(SAMPLE_DOCUMENTS, embedding)

    # 先手动获取查询向量，再直接用向量搜索
    query = "RAG 技术"
    query_vector = embedding.embed_query(query)

    results = vs.similarity_search_by_vector(query_vector, k=2)
    print(f"查询向量来源: '{query}'，返回 top 2:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.page_content}")


def demo_as_retriever():
    """Demo 7: as_retriever() 转换为 Retriever"""
    print("\n" + "=" * 60)
    print("Demo 7: as_retriever()")
    print("=" * 60)

    embedding = get_embedding()
    vs = InMemoryVectorStore.from_documents(SAMPLE_DOCUMENTS, embedding)

    # 转换为 Retriever，统一了 VectorStore 和其他检索方式的接口
    # search_kwargs 传递给 similarity_search 的参数
    retriever = vs.as_retriever(search_kwargs={"k": 2})

    # Retriever 使用 invoke() 调用，返回 Document 列表
    results = retriever.invoke("Embedding 的作用")
    print(f"Retriever.invoke('Embedding 的作用')，返回 top 2:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.page_content}")

    # Retriever 可以直接接入 LCEL Chain
    print("\n→ as_retriever() 返回 VectorStoreRetriever，可直接用于 LCEL Chain")


def demo_delete():
    """Demo 8: 删除文档"""
    print("\n" + "=" * 60)
    print("Demo 8: delete()")
    print("=" * 60)

    embedding = get_embedding()
    vs = InMemoryVectorStore.from_documents(SAMPLE_DOCUMENTS, embedding)
    print(f"初始文档数: {len(vs.store)}")

    # add_documents 返回新增文档的 ID 列表
    new_ids = vs.add_documents([Document(page_content="待删除的文档")])
    print(f"追加 1 个文档，当前文档数: {len(vs.store)}")

    # delete 通过 ID 删除
    vs.delete(ids=new_ids)
    print(f"删除后文档数: {len(vs.store)}")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    demo_from_documents()
    demo_from_texts()
    demo_add_documents()
    demo_similarity_search()
    demo_similarity_search_with_score()
    demo_similarity_search_by_vector()
    demo_as_retriever()
    demo_delete()
    print("\n✅ 所有 Demo 运行完成")
