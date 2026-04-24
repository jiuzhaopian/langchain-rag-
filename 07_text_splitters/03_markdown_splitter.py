"""
03_markdown_splitter.py - MarkdownHeaderTextSplitter + MarkdownTextSplitter

MarkdownHeaderTextSplitter: 按 Markdown 标题层级切分，保留标题路径作为 metadata。
MarkdownTextSplitter: 按 Markdown 语义边界切分，同时遵守 chunk_size 限制。

MarkdownHeaderTextSplitter 拆分规则（源码逻辑）：
  1. 将 headers_to_split_on 按分隔符长度降序排列（如 "###" 排在 "##" 前面），
     确保先匹配更长的标题标记，避免 "###" 被 "#" 误匹配
  2. 逐行扫描 Markdown 文本，遇到匹配的标题行时，记录当前标题层级 metadata
  3. 标题行之间的内容归入当前 metadata 对应的块
  4. 连续多行同 metadata 的内容自动聚合为一个 Document（而非每行一个）
  5. strip_headers=True（默认）时，标题行本身从 content 中移除，只保留正文
  6. 不受 chunk_size 限制，纯按标题结构切分

MarkdownTextSplitter 拆分规则：
  本质上是 RecursiveCharacterTextSplitter 的子类，默认分隔符为
  ["\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n", "\n\n", "\n", " ", ""]，
  即先按 Markdown 标题切，再按代码块结束、水平线（***/---/___）、段落、换行、空格切。
  同时受 chunk_size 限制。注意分隔符被 is_separator_regex=True 作为正则匹配。

参考文档：
  - MarkdownHeaderTextSplitter: https://reference.langchain.com/python/langchain-text-splitters/markdown/MarkdownHeaderTextSplitter
  - MarkdownTextSplitter: https://reference.langchain.com/python/langchain-text-splitters/markdown/MarkdownTextSplitter
  - 源码：https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/markdown.py

安装：
  pip install langchain-text-splitters
"""

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
)


# ============================================================
# 演示 1：MarkdownHeaderTextSplitter（按标题层级切分）
# ============================================================

def demo_header_splitter():
    """
    MarkdownHeaderTextSplitter 按 Markdown 标题层级切分，
    将标题路径存入 metadata。

    特点：
    - 返回 list[Document]（不是 list[str]）
    - 不受 chunk_size 限制，按文档结构切分
    - 每块自动附带标题层级（h1, h2, h3...）

    典型场景：技术文档、README、Wiki 等 Markdown 格式的文档。
    切分后可以通过 metadata 按章节筛选和过滤。
    """
    print("=== 演示 1：MarkdownHeaderTextSplitter ===")

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
    # TODO
    pass


# ============================================================
# 演示 2：MarkdownTextSplitter（按大小切分 Markdown）
# ============================================================

def demo_text_splitter():
    """
    MarkdownTextSplitter 在 Markdown 语义边界处切分，
    同时遵守 chunk_size 限制。

    特点：
    - 返回 list[str]（不是 Document，没有 metadata）
    - 有 chunk_size 限制，控制块大小
    - 适合需要控制大小的场景

    vs MarkdownHeaderTextSplitter:
      MarkdownTextSplitter:     返回 str, 有 chunk_size, 无 metadata
      MarkdownHeaderTextSplitter: 返回 Document, 无 chunk_size, 有标题 metadata
    """
    print("\n\n=== 演示 2：MarkdownTextSplitter ===")

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
        preview = chunk.replace("\n", " ")
        print(f"  块 {i+1}（{len(chunk)} 字符）: {preview}")


# ============================================================
# 演示 3：两者对比
# ============================================================

def demo_comparison():
    """MarkdownHeaderTextSplitter vs MarkdownTextSplitter 选型建议"""
    print("\n\n=== Markdown 两种 Splitter 对比 ===")
    print("""
| 特性           | MarkdownHeaderTextSplitter | MarkdownTextSplitter |
|----------------|--------------------------|---------------------|
| 返回类型       | Document（有 metadata）    | str（无 metadata）    |
| 切分依据       | 标题层级                   | Markdown 语义 + 字符数 |
| chunk_size     | 不受限制                   | 受 chunk_size 控制    |
| 标题信息       | 保留在 metadata 中         | 不保留                |
| 适用场景       | 按章节检索/过滤             | 控制块大小的通用切分   |

组合使用（推荐）:
  先用 MarkdownHeaderTextSplitter 按标题切分，
  再用 RecursiveCharacterTextSplitter 对长块继续拆分。
    """)


if __name__ == "__main__":
    demo_header_splitter()
    demo_text_splitter()
    demo_comparison()
