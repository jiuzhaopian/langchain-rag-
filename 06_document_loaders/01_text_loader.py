"""
01_text_loader.py - TextLoader 文本文件加载器

TextLoader 是最基础的文档加载器，将本地文本文件加载为 Document 对象。
Document 是 LangChain 的核心数据结构：page_content（文本内容）+ metadata（元数据）。

Document 结构:
  - page_content: str       文本内容
  - metadata: dict          元数据（如 source 文件路径、行号等）

TextLoader 返回单个 Document（一个文件 = 一个 Document）。

参考文档：
  - TextLoader: https://reference.langchain.com/python/langchain-community/document_loaders/text/TextLoader

安装：
  pip install langchain-community
"""

import os

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ============================================================
# 公共：创建示例数据文件
# ============================================================

def create_sample_files():
    """创建文本示例文件"""
    os.makedirs(DATA_DIR, exist_ok=True)

    sample_path = os.path.join(DATA_DIR, "sample.txt")
    if not os.path.exists(sample_path):
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write("LangChain 是一个用于构建 LLM 应用的框架。\n")
            f.write("它提供了多种工具和抽象，用于连接大语言模型与外部数据源。\n")
            f.write("核心概念包括：Models、Prompts、OutputParsers、Chains、Retrievers。\n")

    gbk_path = os.path.join(DATA_DIR, "sample_gbk.txt")
    if not os.path.exists(gbk_path):
        with open(gbk_path, "w", encoding="gbk") as f:
            f.write("这是一个 GBK 编码的示例文件。\n")
            f.write("用于演示 TextLoader 的编码处理能力。\n")

    return sample_path, gbk_path


# ============================================================
# 演示 1：加载单个文本文件
# ============================================================

def demo_load_single():
    """
    加载单个文本文件，返回一个 Document。
    """
    print("=== 演示 1：加载单个文本文件 ===")
    # TODO
    pass


# ============================================================
# 演示 2：Document 对象详解
# ============================================================

def demo_document():
    """
    Document 是 LangChain 的核心数据结构，贯穿整个 RAG 流程：
    Loaders 加载 → Splitters 切分 → VectorStore 存储 → Retriever 检索
    """
    print("\n=== 演示 2：Document 对象详解 ===")

    # 手动创建 Document
    doc = Document(
        page_content="LangChain 是一个用于构建 LLM 应用的框架。",
        metadata={
            "source": "手动创建",
            "author": "demo",
            "page": 1,
        },
    )
    print(f"page_content: {doc.page_content}")
    print(f"metadata: {doc.metadata}")

    # Document 是可变的
    doc.metadata["updated"] = True
    print(f"修改后 metadata: {doc.metadata}")


# ============================================================
# 演示 3：懒加载 lazy_load
# ============================================================

def demo_lazy_load():
    """
    lazy_load() 返回生成器，逐条产出 Document。
    适用于大文件或批量文件，避免一次性全部加载到内存。
    """
    print("\n=== 演示 3：懒加载 lazy_load ===")

    loader = TextLoader(os.path.join(DATA_DIR, "sample.txt"))

    # lazy_load 返回 Iterator[Document]
    for i, doc in enumerate(loader.lazy_load()):
        print(f"第 {i+1} 个文档: {doc.metadata}")
        print(f"  内容: {doc.page_content[:80]}...")


# ============================================================
# 演示 4：编码和错误处理
# ============================================================

def demo_encoding():
    """
    TextLoader 默认 encoding=None，会尝试 'utf-8' 解码，失败则抛 UnicodeDecodeError。
    遇到 GBK/GB2312 等编码文件时，需要手动指定 encoding。
    autodetect_encoding 默认 False，设为 True 可自动检测（需安装 chardet/cchardet）。
    """
    print("\n=== 演示 4：编码处理 ===")

    # 指定编码
    loader_gbk = TextLoader(os.path.join(DATA_DIR, "sample_gbk.txt"), encoding="gbk")
    print(f"GBK 编码加载: {loader_gbk.load()[0].page_content[:30]}...")

    # 自动检测编码（需要 chardet 或 cchardet）
    # loader_auto = TextLoader(os.path.join(DATA_DIR, "sample.txt"), autodetect_encoding=True)

    print("编码参数:")
    print("  encoding=None(默认, 尝试 utf-8) / 'gbk' / 'gb2312' / 'utf-16'")
    print("  autodetect_encoding=False(默认) / True(自动检测, 需 chardet)")


if __name__ == "__main__":
    create_sample_files()
    demo_load_single()
    demo_document()
    demo_lazy_load()
    demo_encoding()
