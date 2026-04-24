"""
03_chroma_vectorstore.py - Chroma 向量数据库

Chroma 是一个开源的嵌入式向量数据库，特点是开箱即用、自带持久化、
支持元数据过滤，非常适合本地开发和中小规模应用。

适用场景：需要持久化、需要元数据过滤、中小规模数据（<100万条）。

参考文档：
  - Chroma VectorStore: https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
  - Chroma 官方: https://www.trychroma.com

安装：
  pip install langchain-chroma langchain-ollama
"""

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    print("请安装依赖: pip install langchain-chroma")
    exit(1)


def get_embeddings():
    return OllamaEmbeddings(model="bge-m3")


# Chroma 距离度量：显式指定 cosine，Chroma默认返回L2
#
# bge-m3（以及 OpenAI、DashScope 等主流模型）输出的向量是归一化的（模长=1）。
# 归一化向量下 L2 和 cosine 排序完全等价（L2² = 2(1-cos)），TopK 结果一模一样，
# 但分数值不同：
#   cosine 下：分数 = 余弦相似度（如 0.6），直观易懂
#   L2 下：经公式转换后分数被压缩（如 0.37），"相关文档才 0.37" 容易困惑
# 因此显式指定 cosine，让分数即可解释、又好看。
CHROMA_COLLECTION_METADATA = {"hnsw:space": "cosine"}


# ============================================================
# 演示 1：基本创建和搜索
# ============================================================

def demo_basic():
    """
    Chroma.from_texts() - 基本用法与 InMemoryVectorStore/FAISS 一致。
    Chroma 默认在内存中运行（不指定 persist_directory）。
    """
    embeddings = get_embeddings()

    texts = [
        "LangChain 是一个用于构建 LLM 应用的框架",
        "Python 是一门流行的编程语言",
        "机器学习是人工智能的一个分支",
        "深度学习是机器学习的子领域",
    ]

    # TODO
    pass


# ============================================================
# 演示 2：元数据过滤
# ============================================================

def demo_metadata_filter():
    """
    Chroma 原生支持按 metadata 过滤（filter 参数接收 dict，按 key-value 精确匹配）。
    注意：三种 VectorStore 都支持 metadata 过滤，但方式不同：
      - Chroma: filter=dict（如 {"lang": "python"}），服务端过滤
      - FAISS: filter=dict 或 Callable，客户端过滤（先搜索 fetch_k 条再过滤）
      - InMemory: filter=Callable（函数式过滤）
    """

    embeddings = get_embeddings()

    documents = [
        Document(page_content="Python 爬虫入门", metadata={"lang": "python", "level": "beginner"}),
        Document(page_content="Java 并发编程", metadata={"lang": "java", "level": "advanced"}),
        Document(page_content="Python 数据分析", metadata={"lang": "python", "level": "intermediate"}),
        Document(page_content="Go 语言微服务", metadata={"lang": "go", "level": "intermediate"}),
        Document(page_content="Python 机器学习", metadata={"lang": "python", "level": "advanced"}),
    ]

    vectorstore = Chroma.from_documents(documents, embeddings, collection_metadata=CHROMA_COLLECTION_METADATA)

    print("=== demo_2: metadata 过滤 ===")

    #TODO
    pass


# ============================================================
# 演示 3：持久化
# ============================================================

def demo_persistence():
    """
    Chroma 自带持久化支持，指定 persist_directory 即可。
    数据自动保存到磁盘，下次加载即可继续使用。
    """
    import tempfile
    import os
    import shutil

    embeddings = get_embeddings()

    texts = ["持久化测试文档一", "持久化测试文档二", "持久化测试文档三"]
    persist_dir = tempfile.mkdtemp()

    # 创建并持久化
    vs1 = Chroma.from_texts(texts, embeddings, persist_directory=persist_dir,
                                 collection_metadata=CHROMA_COLLECTION_METADATA)
    print("=== demo_3: 持久化 ===")
    print(f"创建: {len(texts)} 条文档")
    print(f"保存目录: {persist_dir}")
    print(f"目录内容: {os.listdir(persist_dir)}\n")

    # 加载已有集合
    vs2 = Chroma(persist_directory=persist_dir, embedding_function=embeddings, collection_metadata=CHROMA_COLLECTION_METADATA)
    results = vs2.similarity_search("测试文档", k=2)
    print("加载后搜索 '测试文档':")
    for i, doc in enumerate(results):
        print(f"  [{i+1}] {doc.page_content}")

    # 清理
    vs2.delete_collection()
    shutil.rmtree(persist_dir)

    print()


# ============================================================
# 演示 4：增量添加 + 删除
# ============================================================

def demo_add_delete():
    """
    Chroma 支持直接 add_documents / add_texts 到已有集合，
    也支持 delete 按 id 删除。add 时可以指定 ids，用于后续删除。
    """
    import tempfile
    import shutil

    embeddings = get_embeddings()
    persist_dir = tempfile.mkdtemp()

    # add_documents 时指定 ids
    vectorstore = Chroma.from_documents(
        [Document(page_content="初始文档 A", metadata={"type": "init"}),
         Document(page_content="初始文档 B", metadata={"type": "init"})],
        embeddings,
        ids=["doc_a", "doc_b"],
        persist_directory=persist_dir,
        collection_metadata=CHROMA_COLLECTION_METADATA,
    )

    print("=== demo_4: 增量添加 + 删除 ===")
    print(f"初始: 2 条\n")

    # 增量添加（也指定 ids）
    vectorstore.add_texts(["新增文档 C", "新增文档 D"], ids=["doc_c", "doc_d"])
    results = vectorstore.similarity_search("文档", k=5)
    print(f"添加 2 条后 ({len(results)} 条):")
    for doc in results:
        print(f"  - {doc.page_content}")

    # 按 id 删除
    vectorstore.delete(ids=["doc_a", "doc_b"])
    results = vectorstore.similarity_search("文档", k=5)
    print(f"\n删除 doc_a, doc_b 后 ({len(results)} 条):")
    for doc in results:
        print(f"  - {doc.page_content}")

    # 清理
    vectorstore.delete_collection()
    shutil.rmtree(persist_dir)

    print()


# ============================================================
# 演示 5：InMemory vs FAISS vs Chroma 三者对比
# ============================================================

def demo_comparison():
    """
    三种 VectorStore 的特性对比（代码验证 + 结论）。
    """
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_community.vectorstores import FAISS
    import tempfile
    import shutil

    embeddings = get_embeddings()
    texts = ["LangChain 框架", "PyTorch 框架", "React 框架"]

    # InMemory
    vs_inmem = InMemoryVectorStore.from_texts(texts, embeddings)
    r_inmem = [d.page_content for d in vs_inmem.similarity_search("AI框架", k=2)]

    # FAISS
    vs_faiss = FAISS.from_texts(texts, embeddings)
    r_faiss = [d.page_content for d in vs_faiss.similarity_search("AI框架", k=2)]

    # Chroma
    persist_dir = tempfile.mkdtemp()
    vs_chroma = Chroma.from_texts(texts, embeddings, persist_directory=persist_dir, collection_metadata=CHROMA_COLLECTION_METADATA)
    r_chroma = [d.page_content for d in vs_chroma.similarity_search("AI框架", k=2)]
    vs_chroma.delete_collection()
    shutil.rmtree(persist_dir)

    print("=== demo_5: 三种 VectorStore 搜索结果 ===")
    print(f"InMemory: {r_inmem}")
    print(f"FAISS:     {r_faiss}")
    print(f"Chroma:    {r_chroma}")
    print()
    print("--- 结论 ---")
    print("""
| 特性           | InMemory | FAISS           | Chroma         |
|---------------|----------|-----------------|----------------|
| 持久化         | ❌       | save_local 手动 | ✅ 自动         |
| 元数据过滤     | ✅ Callable | ✅ dict/Callable | ✅ dict 原生   |
| 增量添加       | ✅       | merge_from      | ✅ add_texts   |
| 删除           | ✅ delete(ids) | ✅ delete(ids) | ✅ delete(ids) |
| 大数据性能     | 慢       | ⚡ 快（亿级）    | 中等（百万级）  |
| 适用场景       | 教学/测试 | 生产/高性能     | 开发/中小规模   |
| 安装复杂度     | 零依赖    | faiss-cpu       | langchain-chroma |
""")

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
    demo_metadata_filter()
    demo_persistence()
    demo_add_delete()
    demo_comparison()
