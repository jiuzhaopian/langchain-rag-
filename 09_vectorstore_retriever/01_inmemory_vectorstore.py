"""
01_inmemory_vectorstore.py - InMemoryVectorStore 基础

InMemoryVectorStore 是 LangChain 内置的内存向量数据库，适合教学和小规模场景。
数据存储在内存中，进程结束后丢失，不支持持久化。

核心流程：Embeddings → VectorStore.add_documents() → VectorStore.similarity_search()

参考文档：
  - API 参考: https://reference.langchain.com/python/langchain-core/vectorstores/in_memory/InMemoryVectorStore
  - 源码: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/vectorstores/in_memory.py

安装：
  pip install langchain-core langchain-ollama
"""

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings


# ============================================================
# 公共：Embeddings 实例
# ============================================================

def get_embeddings():
    return OllamaEmbeddings(model="bge-m3")


# ============================================================
# 演示 1：从文本列表创建 VectorStore
# ============================================================

def demo_from_texts():
    """
    InMemoryVectorStore.from_texts() - 最简创建方式
    传入文本列表 + Embeddings，一步完成向量化+存储。
    """
    embeddings = get_embeddings()

    texts = [
        "LangChain 是一个用于构建 LLM 应用的框架",
        "Python 是一门流行的编程语言",
        "机器学习是人工智能的一个分支",
        "深度学习是机器学习的子领域",
    ]
    vectorstore = InMemoryVectorStore.from_texts(texts, embeddings)

    print("=== demo_1: from_texts 创建 VectorStore ===")
    print(f"存入 {len(texts)} 条文本\n")

    # similarity_search - 最基础的相似度搜索
    results = vectorstore.similarity_search("什么是机器学习", k=2)
    for i, doc in enumerate(results):
        print(f"  [{i+1}] {doc.page_content}")

    print()


# ============================================================
# 演示 2：从 Document 列表创建，带 metadata
# ============================================================

def demo_from_documents():
    """
    InMemoryVectorStore.from_documents() - 带 metadata 的创建方式
    Document 可以携带 metadata，搜索时一并返回，方便后续过滤和溯源。
    """
    embeddings = get_embeddings()

    documents = [
        Document(page_content="LangChain 支持 Python 和 JavaScript", metadata={"source": "official", "topic": "framework"}),
        Document(page_content="FastAPI 是 Python 的高性能 Web 框架", metadata={"source": "community", "topic": "web"}),
        Document(page_content="PyTorch 是深度学习的主流框架", metadata={"source": "community", "topic": "ml"}),
        Document(page_content="LangGraph 是 LangChain 的扩展，用于构建 Agent", metadata={"source": "official", "topic": "agent"}),
    ]
    vectorstore = InMemoryVectorStore.from_documents(documents, embeddings)

    print("=== demo_2: from_documents 带 metadata ===")
    results = vectorstore.similarity_search("LangChain 生态", k=2)
    for i, doc in enumerate(results):
        print(f"  [{i+1}] {doc.page_content}")
        print(f"       metadata: {doc.metadata}")

    print()


# ============================================================
# 演示 3：手动添加文档 + 带分数搜索
# ============================================================

def demo_add_and_search_with_score():
    """
    动态添加文档 + similarity_search_with_score() - 返回相似度分数
    分数是余弦相似度，范围 0~1，越大越相似（源码用 sklearn 的 cosine_similarity）。
    """
    embeddings = get_embeddings()

    # 创建空的 VectorStore
    vectorstore = InMemoryVectorStore(embeddings)

    # 先添加一批
    vectorstore.add_texts([
        "苹果公司发布了新款 iPhone",
        "华为推出了鸿蒙操作系统",
        "小米电视性价比很高",
    ])
    print("=== demo_3: 动态添加 + 带分数搜索 ===")
    print("第一批: 3 条\n")

    # 再追加一批
    vectorstore.add_documents(documents = [
        Document(page_content="苹果的 M4 芯片性能强劲", metadata={"source": "tans", "topic": "apple"}),
        Document(page_content="三星 Galaxy 手机拍照出色", metadata={"source": "tans", "topic": "sanx"}),
    ])
    print("追加: 2 条，共 5 条\n")

    # 带分数搜索
    results = vectorstore.similarity_search_with_score("苹果产品", k=3)
    print("搜索 '苹果产品' (余弦相似度，越大越相似):")
    for i, (doc, score) in enumerate(results):
        print(f"  [{i+1}] score={score:.4f} | {doc.page_content}")

    print()


# ============================================================
# 演示 4：按相似度阈值过滤
# ============================================================

def demo_similarity_score_threshold():
    """
    similarity_search_with_score + 手动阈值过滤。
    注意：InMemoryVectorStore 不支持 similarity_score_threshold 参数
    （会抛 NotImplementedError），需手动过滤。

    生产环境建议用 FAISS 或 Chroma，它们原生支持阈值过滤。
    """
    embeddings = get_embeddings()

    texts = [
        "猫是一种常见的宠物",
        "狗是人类最忠诚的朋友",
        "量子计算是未来科技",
        "机器学习改变世界",
    ]
    vectorstore = InMemoryVectorStore.from_texts(texts, embeddings)

    print("=== demo_4: 手动阈值过滤 ===")

    # 手动过滤：分数高于阈值的才保留（余弦相似度，越大越相似）
    threshold = 0.5  # 余弦相似度阈值（越高越严格）
    all_results = vectorstore.similarity_search_with_score("可爱的动物", k=4)
    filtered = [(doc, score) for doc, score in all_results if score >= threshold]

    print(f"搜索 '可爱的动物'，阈值={threshold}（余弦相似度）:")
    for i, (doc, score) in enumerate(filtered):
        print(f"  [{i+1}] score={score:.4f} | {doc.page_content}")
    print(f"\n原始 {len(all_results)} 条 → 过滤后 {len(filtered)} 条")

    print()


# ============================================================
# 演示 5：删除文档
# ============================================================

def demo_delete():
    """
    delete() - 按 document id 删除。
    注意：delete() 接收的是 document ID（不是 page_content）。
    add_texts() / add_documents() 返回 ids 列表，保存这些 id 即可后续删除。
    """
    embeddings = get_embeddings()

    # 用 add_texts 获取 ids
    vectorstore = InMemoryVectorStore(embeddings)
    ids = vectorstore.add_texts(["文档A", "文档B", "文档C"])

    print("=== demo_5: 删除文档 ===")
    print(f"初始: 3 条文档")
    print(f"ids: {ids}")

    results = vectorstore.similarity_search("文档", k=5)
    print(f"搜索结果: {[r.page_content for r in results]}")

    # 通过 id 删除第 2 条（文档B）
    vectorstore.delete([ids[1]])
    results = vectorstore.similarity_search("文档", k=5)
    print(f"删除 id={ids[1]} ('文档B') 后: {[r.page_content for r in results]}")

    print()


if __name__ == "__main__":
    import subprocess
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            capture_output=True, timeout=3
        )
        if result.returncode != 0:
            print("Ollama 未运行，请先启动")
            exit(1)
    except Exception as e:
        print(f"无法连接 Ollama: {e}")
        exit(1)

    demo_from_texts()
    demo_from_documents()
    demo_add_and_search_with_score()
    demo_similarity_score_threshold()
    demo_delete()
