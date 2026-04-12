"""
02_character_and_markdown.py - CharacterTextSplitter + MarkdownHeaderTextSplitter

CharacterTextSplitter: 按单个分隔符切分，简单直接。
MarkdownHeaderTextSplitter: 按 Markdown 标题层级切分，保留标题作为 metadata。

参考文档：
  - CharacterTextSplitter: https://python.langchain.com/docs/how_to/character_text_splitter/
  - MarkdownHeaderTextSplitter: https://python.langchain.com/docs/how_to/markdown_header_metadata/

安装：
  pip install langchain-text-splitters
"""

from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
)


# ============================================================
# 演示 1：CharacterTextSplitter
# ============================================================

def demo_character_splitter():
    """
    CharacterTextSplitter 按单个指定分隔符切分。
    如果分隔符之间的文本超过 chunk_size，不会继续拆分（直接保留）。

    vs RecursiveCharacterTextSplitter:
      CharacterTextSplitter: 只用一个分隔符，块可能超长
      RecursiveCharacterTextSplitter: 递归用多个分隔符，块大小更可控
    """
    print("=== 演示 1：CharacterTextSplitter ===")

    text = "第一段内容比较长，它可能会超过 chunk_size。\n\n第二段内容。\n\n第三段内容。"

    splitter = CharacterTextSplitter(
        separator="\n\n",    # 只用这一个分隔符
        chunk_size=50,       # 每块最大字符数
        chunk_overlap=0,     # 无重叠
    )
    chunks = splitter.split_text(text)

    print(f"分隔符='\\n\\n', chunk_size=50")
    print(f"切分为 {len(chunks)} 个块:")
    for i, chunk in enumerate(chunks):
        print(f"  块 {i+1}（{len(chunk)} 字符）: {chunk[:40]}...")

    print(f"\n注意: 第一段 {len('第一段内容比较长，它可能会超过 chunk_size。')} 字符 > chunk_size=50")
    print("CharacterTextSplitter 不会继续拆分，整段保留")


# ============================================================
# 演示 2：MarkdownHeaderTextSplitter（结构化切分）
# ============================================================

def demo_markdown_header():
    """
    MarkdownHeaderTextSplitter 按 Markdown 标题层级切分，
    将标题路径存入 metadata。

    典型场景：技术文档、README、Wiki 等 Markdown 格式的文档。
    切分后可以通过 metadata 按章节筛选和过滤。
    """
    print("\n\n=== 演示 2：MarkdownHeaderTextSplitter ===")

    md_text = """# LangChain 教程

LangChain 是一个用于构建 LLM 应用的框架。

## 第一章：基础概念

Models、Prompts 和 Chains 是 LangChain 的三大核心组件。

### 1.1 Models

Chat Models 是 LangChain 1.x 的唯一推荐模型接口。

### 1.2 Prompts

PromptTemplate 和 ChatPromptTemplate 用于管理提示词。

## 第二章：RAG 系统

RAG = 文档加载 + 文本切分 + 向量存储 + 检索 + 生成。
"""

    # headers_to_split_on: 指定要识别的标题层级和对应的 metadata key
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
    )

    # split_text: str -> list[Document]（注意：不是 list[str]）
    docs = splitter.split_text(md_text)

    print(f"切分为 {len(docs)} 个块:")
    for i, doc in enumerate(docs):
        content_preview = doc.page_content[:60].replace("\n", " ")
        print(f"\n  块 {i+1}:")
        print(f"    metadata: {doc.metadata}")
        print(f"    content:  {content_preview}...")

    print("\n特点:")
    print("  - 每块自动附带标题层级作为 metadata（h1, h2, h3）")
    print("  - 按标题层级切分，保持语义完整性")
    print("  - 不受 chunk_size 限制，按文档结构切分")


# ============================================================
# 演示 3：MarkdownTextSplitter（按大小切分 Markdown）
# ============================================================

def demo_markdown_text():
    """
    MarkdownTextSplitter 在 Markdown 语义边界处切分，
    同时遵守 chunk_size 限制。
    适合需要控制块大小的 Markdown 文档。

    返回 list[str]（不是 Document）。
    """
    print("\n\n=== 演示 3：MarkdownTextSplitter ===")

    md_text = """# Title

Long paragraph about LangChain framework and its various components including models, prompts, and chains.

## Section 1

More content about section 1, discussing the basics of LangChain.

## Section 2

More content about section 2, covering advanced topics and RAG systems.
"""

    splitter = MarkdownTextSplitter(chunk_size=80, chunk_overlap=0)
    chunks = splitter.split_text(md_text)

    print(f"chunk_size=80, 切分为 {len(chunks)} 个文本块:")
    for i, chunk in enumerate(chunks):
        preview = chunk[:50].replace("\n", " ")
        print(f"  块 {i+1}（{len(chunk)} 字符）: {preview}...")

    print("\nvs MarkdownHeaderTextSplitter:")
    print("  MarkdownTextSplitter:     返回 str, 有 chunk_size 限制")
    print("  MarkdownHeaderTextSplitter: 返回 Document, 按标题切分，无 chunk_size")


# ============================================================
# 演示 4：CharacterTextSplitter vs RecursiveCharacterTextSplitter
# ============================================================

def demo_comparison():
    """
    两者对比：何时用哪个？
    """
    print("\n\n=== CharacterTextSplitter vs RecursiveCharacterTextSplitter ===")
    print("""
| 特性           | CharacterTextSplitter  | RecursiveCharacterTextSplitter |
|----------------|----------------------|-------------------------------|
| 分隔符         | 单个                  | 多个（递归）                   |
| 块大小控制     | 块可能超长             | 严格控制在 chunk_size 内       |
| 切分质量       | 一般                  | 好（优先自然断点）              |
| 适用场景       | 已有天然分隔符的文本   | 通用，特别是中文/自然语言       |
| 推荐度         | 了解即可               | **首选默认**                   |

结论：日常使用选 RecursiveCharacterTextSplitter 即可。
    """)


if __name__ == "__main__":
    demo_character_splitter()
    demo_markdown_header()
    demo_markdown_text()
    demo_comparison()
