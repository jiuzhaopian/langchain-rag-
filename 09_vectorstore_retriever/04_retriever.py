"""
04_retriever.py - Retriever 体系

VectorStore 擅长存储和搜索，而 Retriever 是 LangChain 的标准检索接口。
VectorStore.as_retriever() 将 VectorStore 包装为 Retriever，统一检索 API。

Retriever 的核心价值：
  - 统一接口：所有 Retriever 都实现 invoke() 方法，返回 List[Document]
  - 可组合：Retriever 可以像 LCEL 组件一样串联
  - 多种搜索模式：similarity / MMR / similarity_score_threshold

参考文档：
  - Retriever 接口: https://reference.langchain.com/python/langchain-core/retrievers/BaseRetriever
  - VectorStore 基类: https://reference.langchain.com/python/langchain-core/vectorstores/base/VectorStore

安装：
  pip install langchain-core langchain-ollama
"""

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


def get_embeddings():
    return OllamaEmbeddings(model="bge-m3")


# ============================================================
# 准备：创建一个共享的 VectorStore
# ============================================================

def create_vectorstore():
    embeddings = get_embeddings()
    documents = [
        Document(page_content="Python 是一门广泛使用的高级编程语言", metadata={"topic": "python"}),
        Document(page_content="Java 是企业级应用开发的主流语言", metadata={"topic": "java"}),
        Document(page_content="Python 的 NumPy 库用于科学计算", metadata={"topic": "python", "lib": "numpy"}),
        Document(page_content="Java 的 Spring 框架用于后端开发", metadata={"topic": "java", "lib": "spring"}),
        Document(page_content="机器学习是人工智能的核心技术", metadata={"topic": "ml"}),
        Document(page_content="深度学习使用神经网络处理复杂任务", metadata={"topic": "ml"}),
        Document(page_content="Python 是机器学习最常用的编程语言", metadata={"topic": "python", "ml": True}),
        Document(page_content="Python 简单易用", metadata={"topic": "python", "ml": True}),
        Document(page_content="Java 也可以用于机器学习，如 Deeplearning4j", metadata={"topic": "java", "ml": True}),
    ]
    return InMemoryVectorStore.from_documents(documents, embeddings)


# ============================================================
# 演示 1：similarity（默认，纯相似度）
# ============================================================

def demo_similarity():
    """
    search_type="similarity" - 默认模式，按相似度排序返回 Top-K。
    最简单直接，适合大多数场景。
    """
    vs = create_vectorstore()

    print("=== demo_1: similarity（纯相似度） ===")
    print("search_type='similarity', k=3\n")

    #TODO
    pass


# ============================================================
# 演示 2：MMR（最大边际相关性）
# ============================================================

def demo_mmr():
    """
    search_type="mmr" - 最大边际相关性（Maximal Marginal Relevance）。
    平衡「相关性」和「多样性」，避免返回内容高度重复的结果。

    通俗理解：假设你问"Python编程"，纯相似度搜索可能返回：
      1. Python是广泛使用的高级编程语言
      2. Python是机器学习最常用的编程语言
      3. Python的NumPy库用于科学计算
    三条都说Python，内容高度重复。MMR会尽量选不同角度的结果：
      1. Python是广泛使用的高级编程语言  （最相关）
      2. Java也可以用于机器学习  （稍远但提供不同视角）
      3. 机器学习是人工智能的核心技术  （又换个角度）

    关键参数（基于 VectorStore.max_marginal_relevance_search 源码）：
      fetch_k: 先从向量索引取多少条候选文档传给 MMR 算法（默认 20）
      k: 最终返回多少条（默认 4）
      lambda_mult: 0~1 之间，控制多样性程度。
                   0 = 最大多样性（尽量选不同的），
                   1 = 最小多样性（纯相似度，退化为 similarity）
                   默认 0.5
    """
    vs = create_vectorstore()

    print("=== demo_2: MMR（最大边际相关性） ===")

    # 对比：same query, similarity vs MMR
    retriever_sim = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    retriever_mmr = vs.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 8})

    query = "Python 机器学习"
    r_sim = retriever_sim.invoke(query)
    r_mmr = retriever_mmr.invoke(query)

    print(f"查询: '{query}'\n")
    print("similarity 结果（按相似度）:")
    for i, doc in enumerate(r_sim):
        print(f"  [{i+1}] {doc.page_content}")

    print("\nMMR 结果（平衡相关+多样）:")
    for i, doc in enumerate(r_mmr):
        print(f"  [{i+1}] {doc.page_content}")

    print("\n--- 观察 ---")
    print("MMR 会尽量选择与已选结果不重复的内容，避免全是 Python 机器学习相关")
    print("可能引入 Java 或纯 ML 的文档，提高结果覆盖面")

    print()


# ============================================================
# 演示 3：MMR lambda_mult 调参
# ============================================================

def demo_mmr_lambda():
    """
    lambda_mult 控制 MMR 的多样性程度（源码：max_marginal_relevance_search 参数）。
    取值 0~1：0 = 最大多样性，1 = 最小多样性（纯相似度）。
    """
    vs = create_vectorstore()

    print("=== demo_3: MMR lambda_mult 调参 ===")

    query = "编程语言"
    for lam in [0.0, 0.5, 1.0]:
        retriever = vs.as_retriever(
            search_type="mmr", search_kwargs={"k": 3, "fetch_k": 8, "lambda_mult": lam}
        )
        results = retriever.invoke(query)
        topics = [doc.metadata.get("topic", "?") for doc in results]
        print(f"lambda_mult={lam:.1f} → topics={topics}")
        for doc in results:
            print(f"  - {doc.page_content}")
        print()

    print("--- 结论 ---")
    print("lambda_mult=0.0: 最大多样性，尽量选不同主题的文档")
    print("lambda_mult=0.5: 平衡（默认）")
    print("lambda_mult=1.0: 纯相似度，退化为 similarity 模式")

    print()


# ============================================================
# 演示 4：similarity_score_threshold（带阈值过滤）
# ============================================================

def demo_score_threshold():
    """
    search_type="similarity_score_threshold" - 带相似度阈值过滤。
    只返回分数高于阈值的结果，低于阈值的被过滤掉。

    注意：InMemoryVectorStore 不支持此模式（NotImplementedError）。
    FAISS 和 Chroma 支持。

    score 含义（基于源码 _select_relevance_score_fn）：
      FAISS 默认 EUCLIDEAN_DISTANCE，通过 _euclidean_relevance_score_fn
        转为 1.0 - distance / sqrt(2)，输出 0~1 相似度分数（越大越相似）
      Chroma 默认 cosine，通过 _cosine_relevance_score_fn
        转为 1.0 - distance，输出 0~1 相似度分数（越大越相似）
    所以两者的 score_threshold 都是 0~1 范围的相似度，越大越相似。
    """
    try:
        from langchain_community.vectorstores import FAISS
    except ImportError:
        print("请安装依赖: pip install faiss-cpu langchain-community")
        return

    embeddings = get_embeddings()
    texts = [
        "猫是一种常见的宠物",
        "狗是人类最忠诚的朋友",
        "量子计算是前沿科技",
        "深度学习改变世界",
        "今天天气真好",
    ]
    vs = FAISS.from_texts(texts, embeddings)

    print("=== demo_4: similarity_score_threshold ===")
    print("注意：score_threshold 是 0~1 相似度分数，越大越相似\n")

    query = "可爱的动物"
    for threshold in [0.5, 0.7, 0.9]:
        retriever = vs.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": threshold},
        )
        results = retriever.invoke(query)
        print(f"阈值={threshold}: 返回 {len(results)} 条")
        for doc in results:
            print(f"  - {doc.page_content}")
        print()

    print("--- 观察 ---")
    print("阈值越大 → 过滤越严格 → 返回越少（但质量越高）")
    print("阈值越小 → 过滤越宽松 → 返回越多（可能混入不相关的）")

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

    demo_similarity()
    demo_mmr()
    demo_mmr_lambda()
    demo_score_threshold()
