"""
04_html_splitter.py - HTML 切分器（HTMLSectionSplitter / HTMLHeaderTextSplitter / HTMLSemanticPreservingSplitter）

langchain-text-splitters 提供三个 HTML 切分器：

1. HTMLSectionSplitter:    按标题标签切分，保留标题为 metadata，返回 Document
2. HTMLHeaderTextSplitter: 按标题标签切分，保留层级路径为 metadata，支持 return_each_element
3. HTMLSemanticPreservingSplitter (Beta): 语义保持切分，支持 chunk_size，保留链接/图片/视频等

参考文档：
  - HTMLSectionSplitter: https://reference.langchain.com/python/langchain-text-splitters/html/HTMLSectionSplitter
  - HTMLHeaderTextSplitter: https://reference.langchain.com/python/langchain-text-splitters/html/HTMLHeaderTextSplitter
  - HTMLSemanticPreservingSplitter: https://reference.langchain.com/python/langchain-text-splitters/html/HTMLSemanticPreservingSplitter
  - 源码：https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/html.py

安装：
  pip install langchain-text-splitters
  # HTMLSectionSplitter / HTMLSemanticPreservingSplitter 额外需要：
  # pip install lxml bs4
"""

from langchain_text_splitters import HTMLHeaderTextSplitter

try:
    from langchain_text_splitters import HTMLSectionSplitter
    _HAS_LXML = True
except ImportError:
    _HAS_LXML = False


HTML_TEXT = """
<html>
<body>
    <h1>LangChain 教程</h1>
    <p>LangChain 是一个用于构建 LLM 应用的框架。</p>

    <h2>第一章：基础概念</h2>
    <p>Models、Prompts 和 Chains 是三大核心组件。</p>

    <h3>1.1 Models</h3>
    <p>Chat Models 是 LangChain 1.x 的唯一推荐模型接口。</p>

    <h3>1.2 Prompts</h3>
    <p>PromptTemplate 和 ChatPromptTemplate 用于管理提示词。</p>

    <h2>第二章：RAG 系统</h2>
    <p>RAG = 文档加载 + 文本切分 + 向量存储 + 检索 + 生成。</p>
</body>
</html>
"""


# ============================================================
# 演示 1：HTMLHeaderTextSplitter（按标题层级切分，保留层级路径）
# ============================================================

def demo_header_splitter():
    """
    HTMLHeaderTextSplitter 按 HTML 标题标签切分，
    将标题路径存入 metadata（类似 MarkdownHeaderTextSplitter）。

    特点：
    - 返回 list[Document]，每块带标题路径 metadata
    - return_each_element=True 时，每个 HTML 元素独立成块
    - 不受 chunk_size 限制，按文档结构切分

    适合：网页内容的结构化提取。
    """
    print("=== 演示 1：HTMLHeaderTextSplitter ===")

    splitter = HTMLHeaderTextSplitter(
        headers_to_split_on=[
            ("h1", "h1"),
            ("h2", "h2"),
            ("h3", "h3"),
        ],
    )
    docs = splitter.split_text(HTML_TEXT)

    print(f"切分为 {len(docs)} 个块:")
    for i, doc in enumerate(docs):
        content_preview = doc.page_content.replace("\n", " ").strip()
        print(f"\n  块 {i+1}:")
        print(f"    metadata: {doc.metadata}")
        print(f"    content:  {content_preview}")


# ============================================================
# 演示 2：HTMLSectionSplitter（按标题标签切分）
# ============================================================

def demo_section_splitter():
    """
    HTMLSectionSplitter 按 HTML 标题标签切分，
    将标题文本存入 metadata。

    和 HTMLHeaderTextSplitter 的区别：
    - HTMLSectionSplitter: metadata 存单个标题值（如 {"h1": "标题"}）
    - HTMLHeaderTextSplitter: metadata 存完整标题路径（如 {"h1": "标题", "h2": "子标题"}）
    - HTMLSectionSplitter 需要 lxml + bs4

    需要：pip install lxml bs4
    """
    print("\n\n=== 演示 2：HTMLSectionSplitter ===")
    if not _HAS_LXML:
        print("缺少依赖: lxml 和 bs4")
        print("安装命令: pip install lxml bs4")
        return

    splitter = HTMLSectionSplitter(
        headers_to_split_on=[
            ("h1", "h1"),
            ("h2", "h2"),
            ("h3", "h3"),
        ],
    )
    try:
        docs = splitter.split_text(HTML_TEXT)
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("安装命令: pip install lxml bs4")
        return

    print(f"切分为 {len(docs)} 个块:")
    for i, doc in enumerate(docs):
        content_preview = doc.page_content.replace("\n", " ").strip()
        print(f"\n  块 {i+1}:")
        print(f"    metadata: {doc.metadata}")
        print(f"    content:  {content_preview}")


# ============================================================
# 演示 3：三个 HTML Splitter 对比
# ============================================================

def demo_comparison():
    """三个 HTML Splitter 的对比"""
    print("\n\n=== 三个 HTML Splitter 对比 ===")
    print("""
| 特性         | HTMLHeaderTextSplitter      | HTMLSectionSplitter       | HTMLSemanticPreservingSplitter (Beta) |
|-------------|---------------------------|-------------------------|-------------------------------------|
| 返回类型     | Document（带层级 metadata）  | Document（带标题 metadata）| Document（带 metadata + 语义保持）     |
| 切分依据     | 标题标签                   | 标题标签                  | 标题标签 + chunk_size                 |
| chunk_size   | 不支持                     | 不支持                    | 支持（max_chunk_size）                |
| 层级路径     | 保留（h1→h2→h3）           | 仅当前标题                | 保留                                 |
| 保留链接/图片 | 不保留                     | 不保留                    | 可选保留（preserve_links/images/...） |
| 额外依赖     | 无                         | lxml + bs4               | lxml + bs4                           |
| 稳定性       | 稳定                       | 稳定                      | Beta（API 可能变更）                   |

选型建议：
  - 一般网页 → HTMLHeaderTextSplitter（无额外依赖，层级 metadata 最清晰）
  - 需要标题标签匹配 → HTMLSectionSplitter（依赖 lxml，标签匹配更灵活）
  - 需要控制块大小 + 保留媒体 → HTMLSemanticPreservingSplitter（Beta，功能最全）
    """)


if __name__ == "__main__":
    demo_header_splitter()
    demo_section_splitter()
    demo_comparison()
