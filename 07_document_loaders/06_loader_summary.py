"""
06_loader_summary.py - Document Loaders 总结 + Document 数据流

Document 是 LangChain 的核心数据结构，贯穿整个数据处理流程：
  Document Loaders → Text Splitters → VectorStore → Retriever → Chain

本文件汇总 Loader 知识，不加载外部文件。
"""

from langchain_core.documents import Document


# ============================================================
# 演示 1：Document 数据结构
# ============================================================

def demo_document_structure():
    """
    Document = page_content + metadata
    """
    print("=== Document 数据结构 ===")
    print("""
        Document
          |- page_content: str    # 文本内容
          |- metadata: dict       # 元数据
          |- type: str = "Document"
            """)

    doc = Document(
        page_content="LangChain 是 LLM 应用开发框架",
        metadata={"source": "demo", "page": 1},
    )
    print(f"page_content: {doc.page_content}")
    print(f"metadata: {doc.metadata}")


# ============================================================
# 演示 2：Loader 接口统一
# ============================================================

def demo_loader_interface():
    """
    所有 Loader 共享统一接口：
    - load()         → List[Document]
    - lazy_load()    → Iterator[Document]
    - alazy_load()   → AsyncIterator[Document]
    """
    print("\n=== Loader 统一接口 ===")
    print("""
所有 Loader 实现同一接口:

  loader.load()           # 一次性加载全部，返回 List[Document]
  loader.lazy_load()      # 懒加载，返回 Iterator[Document]（逐条产出）
  loader.alazy_load()     # 异步懒加载，返回 AsyncIterator[Document]

选择建议:
  - 小文件 / 少量文件 → load() 简单直接
  - 大文件 / 大量文件 → lazy_load() 节省内存
  - 异步场景          → alazy_load()
    """)


# ============================================================
# 演示 3：常用 Loader 速查
# ============================================================

def demo_loader_cheatsheet():
    """
    常用 Loader 速查表。
    """
    print("\n=== 常用 Loader 速查 ===")
    print("""
| Loader            | 来源           | 安装依赖            | 说明 |
|-------------------|----------------|--------------------|------|
| TextLoader        | langchain-community | 无         | 纯文本文件 |
| DirectoryLoader   | langchain-community | 无         | 目录批量加载 |
| CSVLoader         | langchain-community | 无         | CSV 表格 |
| JSONLoader        | langchain-community | jq         | JSON（支持 jq 表达式）|
| PyPDFLoader       | langchain-community | pypdf      | PDF（每页一个 Document）|
| PyPDFium2Loader   | langchain-community | pypdfium2  | PDF（更快，基于 C++）|
| UnstructuredLoader| langchain-community | unstructured| 通用（HTML/DOCX/PPT 等）|
| WebBaseLoader     | langchain-community | bs4        | 网页（基于 BeautifulSoup）|
| PythonLoader      | langchain-community | 无         | Python 源码文件 |

官方文档：
  TextLoader:         https://reference.langchain.com/python/langchain-community/document_loaders/text/TextLoader
  DirectoryLoader:    https://reference.langchain.com/python/langchain-community/document_loaders/directory/DirectoryLoader
  CSVLoader:          https://docs.langchain.com/oss/python/integrations/document_loaders/csv
  JSONLoader:         https://docs.langchain.com/oss/python/integrations/document_loaders/json
  PyPDFLoader:        https://docs.langchain.com/oss/python/integrations/document_loaders/pypdfloader
  UnstructuredLoader: https://docs.langchain.com/oss/python/integrations/document_loaders/unstructured_file
  WebBaseLoader:      https://docs.langchain.com/oss/python/integrations/document_loaders/web_base
  PythonLoader:       https://docs.langchain.com/oss/python/integrations/document_loaders/python

导入方式统一:
  from langchain_community.document_loaders import TextLoader, CSVLoader, ...
    """)


# ============================================================
# 演示 4：完整数据流
# ============================================================

def demo_data_flow():
    """
    Document 数据流：Loaders → Splitters → VectorStore → Retriever → Chain
    """
    print("\n=== Document 数据流（RAG 全流程）===")
    print("""
步骤 1: Document Loaders  -- 加载原始文档
  PDF/TXT/CSV/HTML → List[Document]

步骤 2: Text Splitters   -- 切分为小块
  List[Document] → List[Document] (切分后)

步骤 3: Embeddings        -- 转为向量
  Document.page_content → List[float] (向量)

步骤 4: VectorStore       -- 存储向量
  (向量, Document) → VectorStore

步骤 5: Retriever         -- 检索相关文档
  查询 → VectorStore.search() → List[Document] (最相关的)

步骤 6: Chain / Agent     -- 组装 Prompt + LLM 调用
  (检索结果 + 用户问题) → Prompt → LLM → 回答

本章（07）学习步骤 1，下一章（08）学习步骤 2。
    """)


if __name__ == "__main__":
    demo_document_structure()
    demo_loader_interface()
    demo_loader_cheatsheet()
    demo_data_flow()
