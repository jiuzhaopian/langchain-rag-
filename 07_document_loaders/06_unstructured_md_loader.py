"""
06_unstructured_md_loader.py - Markdown 文件加载（TextLoader vs UnstructuredMarkdownLoader）

Markdown 本质是纯文本，TextLoader 可以直接加载，保留完整的 markdown 语法（标题、列表、代码块等）。
如果需要将 markdown 解析为结构化元素（按标题分段等），可用 UnstructuredMarkdownLoader（需安装 unstructured）。

参考文档：
  TextLoader:  https://reference.langchain.com/python/langchain-community/document_loaders/text/TextLoader
  UnstructuredMarkdownLoader: https://reference.langchain.com/python/langchain-community/document_loaders/UnstructuredMarkdownLoader
"""

import os

from langchain_community.document_loaders import TextLoader

# 尝试导入 UnstructuredMarkdownLoader（可选依赖）
try:
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
    HAS_UNSTRUCTURED = True
except ImportError:
    HAS_UNSTRUCTURED = False

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(DATA_DIR, "data", "sample.md")


# ============================================================
# 演示 1：TextLoader 加载 Markdown（保留原始语法）
# ============================================================

def demo_textloader_markdown():
    """
    TextLoader 直接加载 .md 文件，保留完整的 markdown 格式。
    适用场景：需要保留原始格式做后续处理（如 chunking、检索）。
    """
    print("=== 演示 1：TextLoader 加载 Markdown ===")
    loader = TextLoader(DATA_FILE)
    docs = loader.load()

    print(f"文档数: {len(docs)}")
    print(f"元数据: {docs[0].metadata}")
    print(f"\n--- 内容预览 ---")
    print(docs[0].page_content)
    print(f"\n是否保留 markdown 语法: {'**LLM 应用**' in docs[0].page_content}")


# ============================================================
# 演示 2：UnstructuredMarkdownLoader（结构化解析，可选）
# ============================================================

def demo_unstructured_markdown():
    """
    UnstructuredMarkdownLoader 将 markdown 解析为结构化元素（Title、NarrativeText、ListItem 等），
    自动按标题分段，适合需要结构化处理的场景。
    需要安装: pip install unstructured markdown
    """
    print("\n=== 演示 2：UnstructuredMarkdownLoader（结构化解析）===")
    if not HAS_UNSTRUCTURED:
        print("未安装 unstructured，跳过此 demo。")
        print("安装命令: pip install unstructured markdown")
        return

    try:
        loader = UnstructuredMarkdownLoader(DATA_FILE)
        docs = loader.load()
    except ModuleNotFoundError as e:
        missing = str(e).split("'")[-2] if "'" in str(e) else str(e)
        print(f"缺少依赖: {missing}")
        print("安装命令: pip install unstructured markdown")
        return

    print(f"文档数: {len(docs)}")
    print(f"元数据: {docs[0].metadata}")
    print(f"\n--- 内容预览 ---")
    print(docs[0].page_content[:500])


# ============================================================
# 演示 3：TextLoader vs UnstructuredMarkdownLoader 对比
# ============================================================

def demo_comparison():
    """
    两种 Loader 的核心区别：
    - TextLoader: 原样返回，保留 markdown 语法符号（**、##、- 等）
    - UnstructuredMarkdownLoader: 解析为纯文本，去掉语法符号，按元素类型分段
    """
    print("\n=== 演示 3：两种 Loader 对比 ===")
    if not HAS_UNSTRUCTURED:
        print("未安装 unstructured，仅展示 TextLoader 输出。")
        loader = TextLoader(DATA_FILE)
        docs = loader.load()
        print(f"\nTextLoader 输出（前 300 字）:")
        print(docs[0].page_content[:300])
        return

    # TextLoader
    docs_text = TextLoader(DATA_FILE).load()
    # UnstructuredMarkdownLoader
    try:
        docs_unstruct = UnstructuredMarkdownLoader(DATA_FILE).load()
    except ModuleNotFoundError as e:
        missing = str(e).split("'")[-2] if "'" in str(e) else str(e)
        print(f"缺少依赖: {missing}")
        print("安装命令: pip install unstructured markdown")
        return

    print(f"{'':30} | TextLoader          | UnstructuredMarkdownLoader")
    print(f"{'-'*30}-+-{'-'*20}-+-{'-'*30}")
    print(f"{'文档数':30} | {str(len(docs_text)):20} | {len(docs_unstruct)}")
    print(f"{'保留 markdown 语法':30} | {'是':20} | {'否（纯文本）'}")
    print(f"{'按标题分段':30} | {'否（整体一份）':20} | {'是（多段）'}")
    print(f"{'依赖':30} | {'无':20} | {'unstructured'}")
    print(f"{'适用场景':30} | {'检索/分块后处理':18} | {'结构化提取'}")


if __name__ == "__main__":
    demo_textloader_markdown()
    demo_unstructured_markdown()
    demo_comparison()
