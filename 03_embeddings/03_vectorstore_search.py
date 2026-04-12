"""
03_vectorstore_search.py
VectorStore 语义搜索 - Embeddings 的核心应用场景

完整流程：
  1. 文档 → Embeddings 转为向量 → 存入 VectorStore（索引阶段）
  2. 查询 → Embeddings 转为向量 → 在 VectorStore 中搜索最相似的文档（检索阶段）

VectorStore.as_retriever() 返回一个 Retriever 对象，
Retriever 是 LangChain 中"检索"的统一接口，后续会用在 RAG（检索增强生成）中。

参考文档：
  - Vector store integrations: https://docs.langchain.com/oss/python/integrations/vectorstores

安装：
  pip install langchain-ollama langchain-core
"""

import subprocess
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document


def check_ollama():
    """检查 Ollama 是否运行"""
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            capture_output=True, timeout=3
        )
        return result.returncode == 0
    except Exception:
        return False


def get_embeddings():
    """获取本地 Ollama Embeddings（无需 API Key）"""
    return OllamaEmbeddings(model="bge-m3")


# ============================================================
# 演示 1：基础语义搜索
# ============================================================

def demo_basic_search():
    """
    基础语义搜索：similarity_search
    """
    print("=== 基础语义搜索 ===")

    embeddings = get_embeddings()

    # 准备文档
    documents = [
        "LangChain 是构建上下文感知推理应用的框架",
        "LangGraph 是用于构建有状态多参与者 LLM 应用的库",
        "Mem0 是一个 AI 记忆层，用于给 AI 应用添加长期记忆",
        "Ollama 是一个本地运行大模型的工具",
        "Python 是最流行的 AI 开发语言",
        "Java 是一门面向对象的编程语言",
        "今天天气真好",
    ]

    # 创建 VectorStore 并索引文档
    # InMemoryVectorStore 是内存向量库，适合开发测试，生产环境用 FAISS/Chroma 等

    vectorstore = InMemoryVectorStore(embedding=embeddings)
    vectorstore.add_texts(documents)

    # vectorstore = InMemoryVectorStore.from_texts(
    #     documents,
    #     embedding=embeddings,
    # )
    print(f"已索引 {len(documents)} 条文档")

    # 直接搜索（返回 Document 列表）
    print("\n1.执行相似度搜索:")
    print(f"\n查询: '什么是 LangChain'")

    results:list[Document] = vectorstore.similarity_search("什么是 LangChain", k=2)
    for i, doc in enumerate(results):
        print(f"  结果 {i + 1}: {doc.page_content}")

    print("\n2.执行带分数的搜索（分数是余弦相似度，0~1，越大越相似）")

    results = vectorstore.similarity_search_with_score("AI 框架", k=3)
    for doc, score in results:
        print(f"  [相似度: {score:.4f}] {doc.page_content}")


# ============================================================
# 演示 2：as_retriever 与搜索参数
# ============================================================

def demo_retriever():
    """
    as_retriever() 将 VectorStore 转为 Retriever 接口。

    Retriever 是 LangChain 的标准检索接口，后续 RAG 链中会直接用 Retriever。

    常用参数：
      search_type="similarity"        默认，按向量相似度搜索
      search_type="mmr"               最大边际相关性，平衡相关性和多样性
      search_type="similarity_score_threshold"  带相似度阈值过滤

      search_kwargs={"k": 2}           返回 top-k 条结果
      search_kwargs={"score_threshold": 0.5}     最低相似度阈值
      search_kwargs={"fetch_k": 10}             mmr 时先候选 fetch_k 条，再从中选 k 条
      search_kwargs={"lambda_mult": 0.5}        mmr 多样性参数，0=最大多样性，1=纯相似度
    """
    print("=== Retriever 搜索 ===")

    embeddings = get_embeddings()

    documents = [
        "LangChain 是构建上下文感知推理应用的框架",
        "LangGraph 是用于构建有状态多参与者 LLM 应用的库",
        "LlamaIndex 是一个数据框架，用于将大语言模型与外部数据连接",
        "Mem0 是一个 AI 记忆层，用于给 AI 应用添加长期记忆",
        "Ollama 是一个本地运行大模型的工具",
        "vLLM 是一个高吞吐量的 LLM 推理引擎",
        "Python 是最流行的 AI 开发语言",
        "Java 是一门面向对象的编程语言",
    ]

    vectorstore = InMemoryVectorStore.from_texts(documents, embedding=embeddings)

    # 1. 默认：search_type=similarity（纯相似度）
    print("=== 1. similarity（默认，纯相似度） ===")
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )
    results = retriever.invoke("LLM 框架")
    for i, doc in enumerate(results):
        print(f"  {i + 1}. {doc.page_content}")

    # 2. search_type=mmr（最大边际相关性，平衡相关性和多样性）
    print("\n=== 2. mmr（最大边际相关性） ===")
    print("   MMR (Maximal Marginal Relevance) 解决什么问题？")
    print("   普通 similarity 搜索可能返回 3 条几乎一样的结果：")
    print("     1. LangChain 是一个 LLM 框架")
    print("     2. LangChain 可以构建 AI 应用")
    print("     3. LangChain 支持多种模型")
    print("   信息高度重复，浪费返回位。")
    print("   MMR 的做法：不仅看和查询有多相关，还看和已选结果有多不同。")
    print("   好比去自助餐厅，不会拿 3 盘一样的菜，而是尽量选不同种类。")
    print("")
    print("   fetch_k=10  → 先从库中取 10 条候选")
    print("   k=3         → 从 10 条中选出 3 条")
    print("   lambda_mult=0.5  → 相关性和多样性的权重平衡，")
    print("                        0 = 只看多样性，1 = 只看相关性")
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},
    )
    results = retriever.invoke("AI 开发工具和编程语言")
    for i, doc in enumerate(results):
        print(f"  {i + 1}. {doc.page_content}")

    # 3. 手动阈值过滤（InMemoryVectorStore 不支持 similarity_score_threshold）
    print("\n=== 3. 手动阈值过滤 ===")
    print("   InMemoryVectorStore 不支持 similarity_score_threshold")
    print("   替代方案：用 similarity_search_with_score 拿分数，手动过滤")
    results = vectorstore.similarity_search_with_score("量子计算", k=8)
    # 余弦相似度 0~1，越大越相似，阈值 ≥ 0.5 表示保留中等以上相关度
    threshold = 0.5
    filtered = [(doc, score) for doc, score in results if score >= threshold]
    if filtered:
        for i, (doc, score) in enumerate(filtered):
            print(f"  {i + 1}. [相似度: {score:.4f}] {doc.page_content}")
    else:
        print(f"  （无结果，所有文档相似度都低于阈值 {threshold}）")
    print("   注: 生产环境用 FAISS/Chroma 等支持 relevance score 的向量库")





if __name__ == "__main__":
    if not check_ollama():
        print("Ollama 未运行，请先启动")
        exit(1)

    demo_basic_search()
    print("\n" + "=" * 50 + "\n")
    demo_retriever()

