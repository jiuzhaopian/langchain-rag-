"""
02_faiss_vectorstore.py - FAISS 向量数据库

FAISS (Facebook AI Similarity Search) 是 Meta 开源的高性能向量搜索库。
特点是速度快、支持大规模数据（亿级），但纯内存运行，不自带持久化（需手动 save_local/load_local）。

适用场景：数据量大（>10万条）、需要快速检索、不需要复杂的元数据过滤。

参考文档：
  - FAISS VectorStore: https://docs.langchain.com/oss/python/integrations/vectorstores/faiss
  - FAISS 官方: https://github.com/facebookresearch/faiss

安装：
  pip install langchain-community langchain-ollama faiss-cpu

  # Windows / Linux / macOS (Intel & Apple Silicon) 均可直接安装 faiss-cpu
  # 如需 GPU 加速（需 CUDA 环境）：pip install faiss-gpu
"""

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    print("请安装依赖: pip install faiss-cpu langchain-community")
    exit(1)


def get_embeddings():
    return OllamaEmbeddings(model="bge-m3")


# ============================================================
# 演示 1：基本创建和搜索
# ============================================================

def demo_basic():
    """
    FAISS.from_texts() - 与 InMemoryVectorStore 用法几乎一致，
    但底层使用 FAISS 索引，搜索速度快得多。
    """
    embeddings = get_embeddings()

    texts = [
        "LangChain 是一个用于构建 LLM 应用的框架",
        "Python 是一门流行的编程语言",
        "机器学习是人工智能的一个分支",
        "深度学习是机器学习的子领域",
        "自然语言处理是 AI 的重要方向",
    ]
    vectorstore = FAISS.from_texts(texts, embeddings)

    print("=== demo_1: FAISS 基本搜索 ===")
    print(f"存入 {len(texts)} 条文本\n")

    results = vectorstore.similarity_search("人工智能技术", k=3)
    for i, doc in enumerate(results):
        print(f"  [{i+1}] {doc.page_content}")

    print()


# ============================================================
# 演示 2：带 metadata + 带分数搜索
# ============================================================

def demo_with_metadata():
    """
    FAISS 支持存储 metadata，搜索时一并返回。
    也支持按 metadata 过滤：filter 参数接收 dict 或 Callable。

    过滤机制：先向量搜索 fetch_k 条（默认 20），再对结果做 filter 过滤（客户端过滤）， 最后从过滤后的结果中返回 top k 条。
    这意味着如果符合条件的文档没被搜进 fetch_k，就会被漏掉。
    可以通过增大 fetch_k 来降低漏检概率。
    """
    embeddings = get_embeddings()

    documents = [
        Document(page_content="Python 爬虫技术入门", metadata={"category": "python", "level": "beginner"}),
        Document(page_content="Java Spring Boot 实战", metadata={"category": "java", "level": "intermediate"}),
        Document(page_content="Python 数据分析实战", metadata={"category": "python", "level": "intermediate"}),
        Document(page_content="机器学习算法详解", metadata={"category": "ml", "level": "advanced"}),
    ]
    vectorstore = FAISS.from_documents(documents, embeddings)

    # 2a: 基本搜索（不过滤）
    print("=== demo_2a: 带 metadata 搜索（不过滤） ===")
    results = vectorstore.similarity_search_with_score("Python 学习", k=2)
    print("搜索 'Python 学习' (FAISS 默认用 L2 距离，score 越小越相似):")
    for i, (doc, score) in enumerate(results):
        print(f"  [{i+1}] score={score:.4f} | {doc.page_content}")
        print(f"       metadata: {doc.metadata}")
    print()

    # 2b: 按 metadata 过滤
    print("=== demo_2b: metadata 过滤 (category=python) ===")
    results = vectorstore.similarity_search(
        "Python 学习",
        k=2,
        filter={"category": "python"},
        fetch_k=20,  # 先搜 20 条再过滤，减少漏检
    )
    print("过滤条件: category=python")
    for i, doc in enumerate(results):
        print(f"  [{i+1}] {doc.page_content}")
        print(f"       metadata: {doc.metadata}")

    print()


# ============================================================
# 演示 3：增量添加文档（merge_from）
# ============================================================

def demo_merge():
    """
    FAISS 不支持直接 add_texts 到已有索引（与 InMemoryVectorStore 不同）。
    增量添加需要用 merge_from() 合并两个 FAISS 索引。
    """
    embeddings = get_embeddings()

    # 创建第一个索引
    vs1 = FAISS.from_texts(["苹果公司", "谷歌公司"], embeddings)

    # 创建第二个索引
    vs2 = FAISS.from_texts(["微软公司", "亚马逊"], embeddings)

    # 合并
    vs1.merge_from(vs2)

    print("=== demo_3: merge_from 增量添加 ===")
    print(f"vs1: 2 条 → 合并后 {vs1.index.ntotal} 条\n")

    results = vs1.similarity_search("科技公司", k=4)
    for i, doc in enumerate(results):
        print(f"  [{i+1}] {doc.page_content}")

    print()


# ============================================================
# 演示 4：持久化 save_local / load_local
# ============================================================

def demo_persistence():
    """
    FAISS 支持将索引保存到磁盘，后续可加载使用。
    注意：metadata 不随 faiss index 保存，需要额外处理。
    """
    import tempfile
    import os

    embeddings = get_embeddings()

    texts = ["这是第一条文档", "这是第二条文档", "这是第三条文档"]
    vectorstore = FAISS.from_texts(texts, embeddings)

    print("=== demo_4: 持久化 ===")

    # 保存
    save_dir = tempfile.mkdtemp()
    vectorstore.save_local(save_dir)
    print(f"保存到: {save_dir}")
    print(f"文件: {os.listdir(save_dir)}\n")

    # 加载
    loaded_vs = FAISS.load_local(save_dir, embeddings, allow_dangerous_deserialization=True)
    results = loaded_vs.similarity_search("文档", k=2)
    print("加载后搜索 '文档':")
    for i, doc in enumerate(results):
        print(f"  [{i+1}] {doc.page_content}")

    # 清理
    import shutil
    shutil.rmtree(save_dir)

    print()


# ============================================================
# 演示 5：InMemoryVectorStore vs FAISS 搜索结果一致性验证
# ============================================================

def demo_consistency():
    """
    相同数据、相同 Embeddings，InMemoryVectorStore 和 FAISS 搜索结果是否一致？

    注意两者的距离度量不同：
    - FAISS 默认用 L2 距离（IndexFlatL2），score 越小越相似
    - InMemoryVectorStore 用余弦相似度，score 越大越相似

    什么是 L2 距离（欧氏距离）？
      两个向量 a 和 b 的 L2 距离 = sqrt(sum((a_i - b_i)^2))
      即各维度差值的平方和再开根号。值域 [0, +∞)，0 表示完全相同。

    什么是余弦相似度？
      cosine(a, b) = (a·b) / (|a| × |b|)
      衡量向量方向的夹角，不关心长度。值域 [-1, 1]，1 表示方向完全相同。

    两者的排序结果通常一致（最相似的文档不管用哪种度量都排在前面），
    但分数非常接近时可能排序不同。
    """
    from langchain_core.vectorstores import InMemoryVectorStore

    embeddings = get_embeddings()
    texts = [
        "LangChain 是 LLM 应用开发框架",
        "PyTorch 是深度学习框架",
        "React 是前端 UI 框架",
        "FastAPI 是 Python Web 框架",
    ]

    vs_inmem = InMemoryVectorStore.from_texts(texts, embeddings)
    vs_faiss = FAISS.from_texts(texts, embeddings)

    print("=== demo_5: InMemory vs FAISS 一致性 ===")
    query = "AI 框架"
    r1 = [doc.page_content for doc in vs_inmem.similarity_search(query, k=2)]
    r2 = [doc.page_content for doc in vs_faiss.similarity_search(query, k=2)]

    print(f"InMemory: {r1}")
    print(f"FAISS:     {r2}")
    print(f"一致: {'✅' if r1 == r2 else '⚠️ 排序略有差异（不同距离度量，分数非常接近时可能排序不同）'}")

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

    demo_basic()
    demo_with_metadata()
    demo_merge()
    demo_persistence()
    demo_consistency()
