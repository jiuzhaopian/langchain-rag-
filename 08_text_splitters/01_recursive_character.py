"""
01_recursive_character.py - RecursiveCharacterTextSplitter（最常用）

RecursiveCharacterTextSplitter 是 LangChain 最常用的文本切分器。
它按字符数切分，递归尝试多种分隔符，优先在自然断点（段落 > 句子 > 单词）处切分。

原理：
  1. 先尝试用 "\\n\\n"（段落）切分
  2. 如果块仍太大，用 "\\n"（换行）继续切
  3. 还太大用 " "（空格）切
  4. 最后逐字符切

这样保证在语义自然的位置切分，而不是在句子中间硬切。

参考文档：
  - https://python.langchain.com/docs/how_to/recursive_text_splitter/

安装：
  pip install langchain-text-splitters
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ============================================================
# 公共：示例文本
# ============================================================

SAMPLE_TEXT = """LangChain 是一个用于构建 LLM 应用的框架。

它提供了多种工具和抽象，包括 Models（模型）、Prompts（提示词）、Chains（链）和 Retrievers（检索器）。

开发者可以使用 LangChain 快速构建 RAG 系统、智能代理和各种 AI 应用。

RAG（Retrieval Augmented Generation）是一种结合检索和生成的技术。

它通过检索相关文档来增强 LLM 的回答质量，减少幻觉问题。

在实际应用中，RAG 系统通常包含文档加载、文本切分、向量存储和检索等步骤。"""


# ============================================================
# 演示 1：基本切分 - split_text
# ============================================================

def demo_split_text():
    """
    split_text: str -> list[str]（纯文本切分，不保留 metadata）
    """
    print("=== 演示 1：split_text（纯文本切分） ===")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,       # 每个块的最大字符数
        chunk_overlap=20,     # 块之间的重叠字符数（保持上下文连贯）
    )

    chunks = splitter.split_text(SAMPLE_TEXT)
    print(f"chunk_size=100, chunk_overlap=20")
    print(f"切分为 {len(chunks)} 个块:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- 块 {i+1}（{len(chunk)} 字符）---")
        print(chunk)


# ============================================================
# 演示 2：带 metadata 切分 - create_documents
# ============================================================

def demo_create_documents():
    """
    create_documents: list[str] -> list[Document]（为每个块附加 metadata）
    """
    print("\n\n=== 演示 2：create_documents（带 metadata） ===")

    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

    docs = splitter.create_documents(
        texts=[SAMPLE_TEXT],
        metadatas=[{"source": "langchain_intro.txt", "author": "demo"}],
    )
    print(f"生成 {len(docs)} 个 Document:")
    for i, doc in enumerate(docs):
        print(f"  块 {i+1}: {doc.metadata} -> {doc.page_content[:40]}...")


# ============================================================
# 演示 3：切分已有 Document - split_documents
# ============================================================

def demo_split_documents():
    """
    split_documents: list[Document] -> list[Document]（保留原 metadata）
    最常用：从 Loader 加载的 Document 直接切分，metadata 原封不动传递。
    """
    print("\n\n=== 演示 3：split_documents（保留 metadata） ===")

    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

    # 模拟从 Loader 加载的 Document
    docs_in = [
        Document(
            page_content=SAMPLE_TEXT,
            metadata={"source": "chapter1.pdf", "page": 1},
        ),
    ]
    docs_out = splitter.split_documents(docs_in)

    print(f"输入: {len(docs_in)} 个 Document")
    print(f"输出: {len(docs_out)} 个 Document（metadata 保留）:")
    for i, doc in enumerate(docs_out):
        print(f"  块 {i+1}: page={doc.metadata['page']}, source={doc.metadata['source']}, "
              f"长度={len(doc.page_content)}")


# ============================================================
# 演示 4：chunk_size 和 chunk_overlap 的影响
# ============================================================

def demo_parameters():
    """
    chunk_size 和 chunk_overlap 是最关键的两个参数。
    """
    print("\n\n=== 演示 4：参数影响对比 ===")

    configs = [
        (200, 0, "大块、无重叠"),
        (100, 20, "中等、有重叠（推荐）"),
        (50, 10, "小块、有重叠"),
    ]

    for size, overlap, desc in configs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
        chunks = splitter.split_text(SAMPLE_TEXT)
        print(f"\n{desc}: chunk_size={size}, overlap={overlap}")
        print(f"  -> {len(chunks)} 个块", end="")
        if chunks:
            print(f", 块大小范围: {min(len(c) for c in chunks)}-{max(len(c) for c in chunks)} 字符")
        else:
            print()


# ============================================================
# 演示 5：自定义分隔符
# ============================================================

def demo_separators():
    """
    默认分隔符: ["\n\n", "\n", " ", ""]
    可以自定义，如中文场景可以用中文句号、逗号作为分隔符。
    """
    print("\n\n=== 演示 5：自定义分隔符 ===")

    print("默认分隔符（递归顺序）: ['\\n\\n', '\\n', ' ', '']")
    print("  先尝试段落切分 -> 再换行 -> 再空格 -> 最后逐字符")

    # 中文场景自定义
    splitter_cn = RecursiveCharacterTextSplitter(
        chunk_size=80,
        chunk_overlap=10,
        separators=["\n\n", "。", "\n", "，", " ", ""],
    )
    chunks = splitter_cn.split_text(SAMPLE_TEXT)
    print(f"\n中文分隔符 ['\\n\\n', '。', '\\n', '，', ' ', '']:")
    print(f"  -> {len(chunks)} 个块")
    for i, chunk in enumerate(chunks[:3]):
        print(f"  块 {i+1}: {chunk[:50]}...")


# ============================================================
# 演示 6：length_function
# ============================================================

def demo_length_function():
    """
    length_function 控制如何计算"长度"。
    默认用 len()（字符数），可以改为 token 数等。
    """
    print("\n\n=== 演示 6：length_function ===")

    print("默认: length_function=len（按字符数计算）")
    print("可自定义: length_function=lambda x: len(tokenizer.encode(x))（按 token 数）")
    print("注意: 改 length_function 时，chunk_size 的含义也随之变化")

    # 示例：用自定义函数（如计算中文字符为 2 倍权重）
    def cn_weighted_len(text):
        """中文字符算 2，英文算 1"""
        return sum(2 if ord(c) > 127 else 1 for c in text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=cn_weighted_len,
    )
    chunks = splitter.split_text(SAMPLE_TEXT)
    print(f"\n中文加权长度（中文=2, 英文=1）:")
    print(f"  -> {len(chunks)} 个块")


if __name__ == "__main__":
    demo_split_text()
    demo_create_documents()
    demo_split_documents()
    demo_parameters()
    demo_separators()
    demo_length_function()
