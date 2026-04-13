"""
03_rag_chain.py - 完整 RAG（检索增强生成）链

RAG 的核心流程：
  文档加载 → 文本分割 → Embedding → VectorStore → Retriever → Chain

本 Demo 展示两种组装方式：
  1. create_retrieval_chain + create_stuff_documents_chain（高层 API）
  2. RunnablePassthrough 手动组装（LCEL 方式，更灵活）

注意：完整运行需要 ZHIPUAI_API_KEY 环境变量和 Ollama bge-m3 服务。
沙箱环境下自动降级，打印 API 用法说明。

参考文档：
  - create_retrieval_chain: https://python.langchain.com/api_reference/langchain/chains/langchain.chains.retrieval.create_retrieval_chain.html
  - create_stuff_documents_chain: https://python.langchain.com/api_reference/langchain/chains/langchain.chains.combine_documents.stuff.create_stuff_documents_chain.html

安装：
  pip install langchain langchain-core langchain-community langchain-text-splitters
"""

import os

import numpy as np
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
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


# 知识库文档
KNOWLEDGE_BASE = [
    "LangChain 是一个开源框架，用于开发由大语言模型（LLM）驱动的应用。",
    "LangChain 的核心组件包括：Model I/O、Retrieval、Agents、Chains、Memory。",
    "Retrieval 组件负责从外部数据源检索相关信息，供 LLM 参考。",
    "VectorStore 是存储文档向量的数据库，常见的有 Chroma、FAISS、Pinecone。",
    "Embedding 模型将文本转换为向量，常用模型有 bge-m3、text-embedding-3-small。",
    "RAG（Retrieval-Augmented Generation）通过检索外部知识来增强 LLM 的回答。",
    "RAG 流程：用户提问 → 检索相关文档 → 将文档和问题一起传给 LLM → 生成回答。",
    "Text Splitter 将长文档切分为小块，常用策略有按字符数、递归字符、Token 数分割。",
    "BM25 是基于词频的检索算法，适合关键词明确的场景。",
    "MMR（Maximal Marginal Relevance）在检索时兼顾相关性和多样性。",
]


# ============================================================
# Demo 函数
# ============================================================

def demo_rag_pipeline_overview():
    """Demo 1: RAG 流程概览（仅演示 API 调用链，不实际调用 LLM）"""
    print("\n" + "=" * 60)
    print("Demo 1: RAG 流程概览")
    print("=" * 60)

    print("""
RAG 完整流程：

  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │  1. 文档加载  │ →  │  2. 文本分割  │ →  │ 3. Embedding │
  │  (Loader)    │    │  (Splitter)  │    │  (向量化)     │
  └──────────────┘    └──────────────┘    └──────────────┘
                                                 │
                                                 ▼
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │  6. 生成回答  │ ←  │ 5. LLM 推理  │ ←  │  4. 检索     │
  │  (Response)  │    │  (Chain)     │    │  (Retriever) │
  └──────────────┘    └──────────────┘    └──────────────┘
""")

    # 演示步骤 1-4（不需要 LLM）
    print("步骤 1-4（不需要 LLM）:")

    # Step 1: 文档加载（这里直接用字符串列表）
    print("  1. 文档加载 → 已准备 {} 条知识".format(len(KNOWLEDGE_BASE)))

    # Step 2: 文本分割
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.create_documents(KNOWLEDGE_BASE)
    print(f"  2. 文本分割 → {len(chunks)} 个文档块")

    # Step 3: Embedding
    embedding = FakeEmbedding()
    print("  3. Embedding → 使用 FakeEmbedding（实际用 Ollama bge-m3）")

    # Step 4: VectorStore + Retriever
    vs = InMemoryVectorStore.from_documents(chunks, embedding)
    retriever = vs.as_retriever(search_kwargs={"k": 3})

    query = "RAG 是什么？"
    docs = retriever.invoke(query)
    print(f"  4. 检索 → 查询 '{query}' 返回 {len(docs)} 条:")
    for i, doc in enumerate(docs, 1):
        print(f"     {i}. {doc.page_content[:50]}...")


def demo_stuff_documents_chain():
    """Demo 2: create_stuff_documents_chain（文档拼接链）

    create_stuff_documents_chain 将多个检索到的文档拼接为一个字符串，
    然后传给 LLM。这是 RAG 中最常用的文档合并策略。

    注意：在 langchain 1.2.x 中，create_retrieval_chain 和
    create_stuff_documents_chain 已移除，推荐直接使用 LCEL 方式组装。
    本 demo 展示等效的 LCEL 实现。
    """
    print("\n" + "=" * 60)
    print("Demo 2: 文档拼接 + 检索链（LCEL 实现）")
    print("=" * 60)

    try:
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        from langchain_community.chat_models import ChatZhipuAI
    except ImportError as e:
        print(f"⚠️  缺少依赖: {e}")
        return

    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        print("⚠️  未设置 ZHIPUAI_API_KEY，跳过完整演示")
        print("\n等效代码（原 create_retrieval_chain + create_stuff_documents_chain）：")
        print("""
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    # stuff 策略：将所有文档拼接到 context 中
    def format_docs(docs):
        return "\\n\\n".join(doc.page_content for doc in docs)

    # LCEL 方式等效于 create_retrieval_chain + create_stuff_documents_chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke("RAG 的流程是什么？")
        """)
        return

    # 完整运行
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        chunks = splitter.create_documents(KNOWLEDGE_BASE)

        try:
            from langchain_community.embeddings import OllamaEmbeddings
            embedding = OllamaEmbeddings(model="bge-m3")
            embedding.embed_query("test")
        except Exception:
            embedding = FakeEmbedding()
            print("⚠️  Ollama 不可用，使用 FakeEmbedding")

        vs = InMemoryVectorStore.from_documents(chunks, embedding)
        retriever = vs.as_retriever(search_kwargs={"k": 3})
        llm = ChatZhipuAI(model="glm-4", api_key=api_key)

        prompt = ChatPromptTemplate.from_template(
            "根据以下上下文回答问题。如果上下文中没有相关信息，请说'我不知道'。\n\n"
            "上下文：\n{context}\n\n"
            "问题：{question}"
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm            | StrOutputParser()
        )

        query = "RAG 的完整流程是什么？"
        print(f"查询: '{query}'")
        print("（LLM 调用中，请稍候...）")
        answer = rag_chain.invoke(query)
        print(f"\n回答: {answer}")

    except Exception as e:
        print(f"⚠️  运行出错: {e}")


def demo_manual_lcel_chain():
    """Demo 3: 用 LCEL（RunnablePassthrough）手动组装 RAG 链"""
    print("\n" + "=" * 60)
    print("Demo 3: LCEL 手动组装 RAG 链")
    print("=" * 60)

    try:
        from langchain_core.runnables import RunnablePassthrough, RunnableParallel
        from langchain_community.chat_models import ChatZhipuAI
    except ImportError as e:
        print(f"⚠️  缺少依赖: {e}")
        return

    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        print("⚠️  未设置 ZHIPUAI_API_KEY，跳过完整演示")
        print("\n代码示例:")
        print("""
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    def format_docs(docs):
        return "\\n\\n".join(doc.page_content for doc in docs)

    # LCEL 管道：检索 → 格式化 → Prompt → LLM → 解析
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke("什么是 Embedding？")
        """)
        return

    # 完整运行
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.output_parsers import StrOutputParser

        # 准备
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        chunks = splitter.create_documents(KNOWLEDGE_BASE)

        try:
            from langchain_community.embeddings import OllamaEmbeddings
            embedding = OllamaEmbeddings(model="bge-m3")
            embedding.embed_query("test")
        except Exception:
            embedding = FakeEmbedding()
            print("⚠️  Ollama 不可用，使用 FakeEmbedding")

        vs = InMemoryVectorStore.from_documents(chunks, embedding)
        retriever = vs.as_retriever(search_kwargs={"k": 3})

        llm = ChatZhipuAI(model="glm-4", api_key=api_key)

        prompt = ChatPromptTemplate.from_template(
            "根据以下上下文回答问题：\n\n上下文：\n{context}\n\n问题：{question}"
        )

        def format_docs(docs):
            """将 Document 列表格式化为纯文本"""
            return "\n\n".join(doc.page_content for doc in docs)

        # LCEL 管道组装
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        query = "什么是 BM25？它和向量检索有什么区别？"
        print(f"查询: '{query}'")
        print("（LLM 调用中，请稍候...）")
        answer = rag_chain.invoke(query)
        print(f"\n回答: {answer}")

    except Exception as e:
        print(f"⚠️  运行出错: {e}")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    demo_rag_pipeline_overview()
    demo_stuff_documents_chain()
    demo_manual_lcel_chain()
    print("\n✅ 所有 Demo 运行完成")
