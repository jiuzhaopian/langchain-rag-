"""
06_splitter_summary.py - Text Splitters 总结 + 选择指南

汇总所有 Text Splitter 的知识，包括选型建议和参数调优。
"""


# ============================================================
# 演示 1：Splitter 选型速查
# ============================================================

def demo_cheatsheet():
    """
    Splitter 选型速查表。
    """
    print("=== Text Splitter 选型速查 ===")
    print("""
| Splitter                        | 切分依据           | 返回类型       | 推荐场景              |
|---------------------------------|-------------------|---------------|----------------------|
| RecursiveCharacterTextSplitter  | 字符数 + 多分隔符递归 | str / Document | **通用首选**            |
| CharacterTextSplitter           | 单个分隔符          | str / Document | 结构清晰的文本（JSON/日志） |
| MarkdownHeaderTextSplitter      | Markdown 标题层级   | Document      | 技术文档/Wiki（按章节检索）  |
| MarkdownTextSplitter            | Markdown 结构+字符数 | str / Document | 需要控制大小的 Markdown    |
| HTMLHeaderTextSplitter          | HTML 标题层级       | Document      | 网页内容（无额外依赖）      |
| HTMLSectionSplitter             | HTML 标题标签       | Document      | 网页内容（需 lxml+bs4）   |
| HTMLSemanticPreservingSplitter  | 标题+chunk_size    | Document      | 需保留链接/图片（Beta）    |
| LatexTextSplitter               | LaTeX 章节/环境     | str / Document | 学术论文/技术报告          |
| TokenTextSplitter               | Token 数           | str / Document | 精确控制 token           |
| PythonCodeTextSplitter          | Python 语法        | str / Document | Python 源码             |

说明：
  - str = split_text() 的返回类型
  - Document = split_documents() 的返回类型，带 metadata
  - 所有 Splitter 的 split_documents() 都返回 Document，部分也支持 split_text() 返回 str

导入:
  from langchain_text_splitters import RecursiveCharacterTextSplitter, ...
    """)


# ============================================================
# 演示 2：三个切分方法对比
# ============================================================

def demo_three_methods():
    """
    每个 Splitter 都有三个方法，使用场景不同：
    """
    print("\n=== 三个切分方法 ===")
    print("""
  split_text(text)            str -> list[str]
    纯文本切分，不涉及 Document。最简单。

  create_documents(texts, metadatas)
    list[str] -> list[Document]
    为每段文本附加 metadata。适合需要标记来源的场景。

  split_documents(documents)
    list[Document] -> list[Document]
    切分已有 Document（保留原 metadata）。
    **最常用**：Loader 加载后直接切分。

典型流程:
  docs = loader.load()                          # 加载
  chunks = splitter.split_documents(docs)       # 切分
  # chunks 的 metadata 自动继承 docs 的 metadata
    """)


# ============================================================
# 演示 3：参数调优指南
# ============================================================

def demo_tuning():
    """
    chunk_size 和 chunk_overlap 的选择建议。
    """
    print("\n=== 参数调优指南 ===")
    print("""
chunk_size（每块大小）:
  太小（<100）: 丢失上下文，语义不完整
  适中（300-1000）: 大多数场景够用
  太大（>2000）: 检索精度下降，噪音增多
  建议起始值: 500

chunk_overlap（重叠大小）:
  作用: 防止重要信息被切断在两个块的边界
  建议: chunk_size 的 10%-20%
  示例: chunk_size=500 -> overlap=50~100

实际建议:
  - 简单 Q&A: chunk_size=500, overlap=50
  - 长文档分析: chunk_size=1000, overlap=100
  - 代码文档: 用 PythonCodeTextSplitter（按语法切）
  - Markdown 文档: 用 MarkdownHeaderTextSplitter（按标题切）
  - HTML 文档: 用 HTMLHeaderTextSplitter（按标题切，无额外依赖）或 HTMLSectionSplitter（需 lxml）

验证方法:
  切分后随机抽查几个块，确认语义是否完整、是否有明显的语义断裂。
    """)


# ============================================================
# 演示 4：完整 RAG 切分流程示例
# ============================================================

def demo_full_pipeline():
    """
    Loaders + Splitters 完整流程。
    """
    print("\n=== 完整 RAG 切分流程 ===")
    print("""
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 步骤 1: 加载文档
loader = TextLoader("./knowledge_base.txt")
docs = loader.load()
# docs = [Document(page_content="...", metadata={"source": "..."})]

# 步骤 2: 切分文档
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(docs)
# chunks = [Document(...), Document(...), ...]
# 每个 chunk 保留了原始 metadata（source 等信息）

# 步骤 3: 后续可以接 Embedding + VectorStore（下个章节）
    """)


if __name__ == "__main__":
    demo_cheatsheet()
    demo_three_methods()
    demo_tuning()
    demo_full_pipeline()
