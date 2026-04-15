"""
04_retriever.py - Retriever 体系

VectorStore 擅长存储和搜索，而 Retriever 是 LangChain 的标准检索接口。
VectorStore.as_retriever() 将 VectorStore 包装为 Retriever，统一检索 API。

Retriever 的核心价值：
  - 统一接口：所有 Retriever 都实现 invoke() 方法，返回 List[Document]
  - 可组合：Retriever 可以像 LCEL 组件一样串联
  - 多种搜索模式：similarity / MMR / similarity_score_threshold

参考文档：
  - Retriever 接口: https://docs.langchain.com/oss/python/langchain/retrievers
  - VectorStore.as_retriever: https://docs.langchain.com/oss/python/langchain/vectorstores

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
    """创建教学用的 VectorStore（多篇关于不同主题的文档）"""
    embeddings = get_embeddings()
    documents = [
        Document(page_content="Python 是一门广泛使用的高级编程语言", metadata={"topic": "python"}),
        Document(page_content="Java 是企业级应用开发的主流语言", metadata={"topic": "java"}),
        Document(page_content="Python 的 NumPy 库用于科学计算", metadata={"topic": "python", "lib": "numpy"}),
        Document(page_content="Java 的 Spring 框架用于后端开发", metadata={"topic": "java", "lib": "spring"}),
        Document(page_content="机器学习是人工智能的核心技术", metadata={"topic": "ml"}),
        Document(page_content="深度学习使用神经网络处理复杂任务", metadata={"topic": "ml"}),
        Document(page_content="Python 是机器学习最常用的编程语言", metadata={"topic": "python", "ml": True}),
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

    retriever = vs.as_retriever(search_type="similarity", k=3)
    results = retriever.invoke("Python 编程")
    for i, doc in enumerate(results):
        print(f"  [{i+1}] {doc.page_content}")
        print(f"       metadata: {doc.metadata}")

    print()


# ============================================================
# 演示 2：MMR（最大边际相关性）
# ============================================================

def demo_mmr():
    """
    search_type="mmr" - 最大边际相关性（Maximal Marginal Relevance）。
    平衡「相关性」和「多样性」，避免返回内容高度重复的结果。

    关键参数：
      fetch_k: 先从 VectorStore 取多少条候选（默认 20）
      k: 最终返回多少条（默认 4）
      lambda_mult: 多样性权重，0=最大多样性，1=纯相似度（默认 0.5）
    """
    vs = create_vectorstore()

    print("=== demo_2: MMR（最大边际相关性） ===")

    # 对比：same query, similarity vs MMR
    retriever_sim = vs.as_retriever(search_type="similarity", k=4)
    retriever_mmr = vs.as_retriever(search_type="mmr", k=4, fetch_k=8)

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
    lambda_mult 控制 MMR 的多样性程度。
    """
    vs = create_vectorstore()

    print("=== demo_3: MMR lambda_mult 调参 ===")

    query = "编程语言"
    for lam in [0.0, 0.5, 1.0]:
        retriever = vs.as_retriever(
            search_type="mmr", k=3, fetch_k=8, lambda_mult=lam
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

    这里用 FAISS 演示。
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

    # FAISS 的分数是 L2 距离（越小越相似），阈值也是距离
    query = "可爱的动物"
    for threshold in [0.5, 1.0, 2.0]:
        retriever = vs.as_retriever(
            search_type="similarity_score_threshold",
            k=5,
            score_threshold=threshold,
        )
        results = retriever.invoke(query)
        print(f"阈值={threshold}: 返回 {len(results)} 条")
        for doc in results:
            print(f"  - {doc.page_content}")
        print()

    print("--- 观察 ---")
    print("阈值越小 → 过滤越严格 → 返回越少（但质量越高）")
    print("阈值越大 → 过滤越宽松 → 返回越多（可能混入不相关的）")

    print()


# ============================================================
# 演示 5：Retriever 的核心价值 - 统一接口 + LCEL 集成
# ============================================================

def demo_lcel_integration():
    """
    Retriever 实现了 Runnable 接口，可以直接接入 LCEL 链。
    演示 Retriever + RunnablePassthrough 的基本组合。
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    vs = create_vectorstore()
    retriever = vs.as_retriever(search_type="similarity", k=3)

    print("=== demo_5: Retriever + LCEL 链 ===")

    # 定义链（只展示结构，不实际调用 LLM）
    prompt = ChatPromptTemplate.from_template(
        "根据以下上下文回答问题：\n\n{context}\n\n问题：{question}"
    )

    # format_docs: 将 Document 列表格式化为文本
    def format_docs(docs):
        return "\n---\n".join(doc.page_content for doc in docs)

    # 构建 RAG 链的结构
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | StrOutputParser()
    )

    print("RAG 链结构：")
    print("  retriever | format_docs → 构建 context")
    print("  RunnablePassthrough() → 透传 question")
    print("  prompt | StrOutputParser → 生成回答")
    print()
    print(f"Retriever 类型: {type(retriever).__name__}")
    print(f"Retriever 是 Runnable: {hasattr(retriever, 'invoke')}")

    # 测试 retriever 部分（不调用 LLM）
    print(f"\n检索 'Python 机器学习':")
    docs = retriever.invoke("Python 机器学习")
    print(f"返回 {len(docs)} 条文档")

    print("\n注：完整 RAG 链需要 LLM，将在下一节 RAG Chain 中实现")

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
    demo_lcel_integration()
