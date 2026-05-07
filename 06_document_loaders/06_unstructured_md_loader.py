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
    # 修复：显式指定 UTF-8 编码
    loader = TextLoader(DATA_FILE, encoding='utf-8')
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
        # 修复：通过 mode="elements" 获取结构化元素，并指定编码
        loader = UnstructuredMarkdownLoader(
            DATA_FILE,
            mode="elements",
            encoding='utf-8'
        )
        docs = loader.load()
    except ModuleNotFoundError as e:
        missing = str(e).split("'")[-2] if "'" in str(e) else str(e)
        print(f"缺少依赖: {missing}")
        print("安装命令: pip install unstructured markdown")
        return
    except Exception as e:
        print(f"加载失败: {e}")
        return

    print(f"文档数: {len(docs)}")
    print(f"元数据示例: {docs[0].metadata if docs else '无'}")
    print(f"\n--- 内容预览（前 3 个元素）---")
    for i, doc in enumerate(docs[:3]):
        print(f"\n[{i+1}] 类型: {doc.metadata.get('category', 'unknown')}")
        print(f"内容: {doc.page_content[:100]}...")


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

    # TextLoader（修复编码）
    docs_text = TextLoader(DATA_FILE, encoding='utf-8').load()
    text_content = docs_text[0].page_content

    print(f"\n{'':30} | TextLoader")
    print(f"{'-'*30}-+-{'-'*40}")
    print(f"{'文档数':30} | {len(docs_text)}")
    print(f"{'保留 markdown 语法':30} | 是")
    print(f"{'按标题分段':30} | 否（整体一份）")
    print(f"{'输出示例':30} | {text_content[:150].replace(chr(10), ' ')}...")

    if HAS_UNSTRUCTURED:
        try:
            docs_unstruct = UnstructuredMarkdownLoader(
                DATA_FILE,
                mode="elements",
                encoding='utf-8'
            ).load()

            print(f"\n{'':30} | UnstructuredMarkdownLoader")
            print(f"{'-'*30}-+-{'-'*40}")
            print(f"{'文档数':30} | {len(docs_unstruct)}")
            print(f"{'保留 markdown 语法':30} | 否（纯文本）")
            print(f"{'按标题分段':30} | 是（多段）")

            # 统计各类型元素
            categories = {}
            for doc in docs_unstruct:
                cat = doc.metadata.get('category', 'unknown')
                categories[cat] = categories.get(cat, 0) + 1
            print(f"{'元素类型分布':30} | {categories}")

        except Exception as e:
            print(f"\nUnstructuredMarkdownLoader 加载失败: {e}")
    else:
        print("\n未安装 unstructured，跳过对比。")
        print("安装命令: pip install unstructured markdown")


def demo_mode_comparison():
    """
    演示 UnstructuredMarkdownLoader 的两种模式：
    - mode="single": 单文档模式，所有内容合并为一个 Document
    - mode="elements": 元素模式，按结构拆分为多个 Document
    """
    print("\n=== 演示 4：UnstructuredMarkdownLoader 模式对比 ===")
    if not HAS_UNSTRUCTURED:
        print("未安装 unstructured，跳过此 demo。")
        print("安装命令: pip install unstructured markdown")
        return

    try:
        # single 模式
        loader_single = UnstructuredMarkdownLoader(
            DATA_FILE,
            mode="single",
            encoding='utf-8'
        )
        docs_single = loader_single.load()
        print(f"\n模式 'single':")
        print(f"  - 文档数: {len(docs_single)}")
        print(f"  - 内容长度: {len(docs_single[0].page_content)} 字符")

        # elements 模式
        loader_elements = UnstructuredMarkdownLoader(
            DATA_FILE,
            mode="elements",
            encoding='utf-8'
        )
        docs_elements = loader_elements.load()
        print(f"\n模式 'elements':")
        print(f"  - 文档数: {len(docs_elements)}")
        print(f"  - 每个元素: 标题、段落、列表项等被拆分")

        # 显示前几个元素
        print(f"\n  前 5 个元素类型:")
        for i, doc in enumerate(docs_elements[:5]):
            cat = doc.metadata.get('category', 'unknown')
            print(f"    [{i+1}] {cat}: {doc.page_content[:50]}...")

    except Exception as e:
        print(f"加载失败: {e}")


if __name__ == "__main__":
    demo_textloader_markdown()
    demo_unstructured_markdown()
    demo_comparison()
    demo_mode_comparison()