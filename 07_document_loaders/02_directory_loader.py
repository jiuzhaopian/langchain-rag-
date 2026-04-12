"""
02_directory_loader.py - DirectoryLoader 目录加载器

DirectoryLoader 递归扫描目录，将匹配的文件批量加载为 Document 列表。
本质是遍历文件 + 对每个文件调用指定的 Loader。

重要：必须指定 loader_cls，否则默认使用 UnstructuredLoader（需要安装 unstructured 包）。
常用: loader_cls=TextLoader, glob="**/*.txt"

参考文档：
  - DirectoryLoader: https://python.langchain.com/docs/integrations/document_loaders/directory/

安装：
  pip install langchain-community
"""

import os
import shutil

from langchain_community.document_loaders import DirectoryLoader, TextLoader


# ============================================================
# 演示 1：基本用法 - 扫描目录
# ============================================================

def demo_basic():
    """
    扫描目录，加载所有 .txt 文件。
    """
    print("=== 演示 1：基本用法 ===")

    # 必须指定 loader_cls，否则默认用 UnstructuredLoader（需要额外安装）
    loader = DirectoryLoader(
        "./docs",
        glob="**/*.txt",
        loader_cls=TextLoader,
    )
    docs = loader.load()

    print(f"加载了 {len(docs)} 个文档:")
    for doc in docs:
        print(f"  [{doc.metadata['source']}] {doc.page_content[:50]}...")


# ============================================================
# 演示 2：glob 模式匹配
# ============================================================

def demo_glob_patterns():
    """
    glob 参数控制匹配规则，支持通配符。
    """
    print("\n=== 演示 2：glob 模式 ===")

    patterns = {
        "*.txt": "仅当前目录的 .txt",
        "**/*.txt": "递归所有子目录的 .txt",
        "*.md": "仅当前目录的 .md",
        "data/*.csv": "data 子目录下的 .csv",
        "**/*.py": "递归所有 .py 文件",
    }

    for pattern, desc in patterns.items():
        print(f"  glob='{pattern}' -> {desc}")


# ============================================================
# 演示 3：懒加载 + silent_errors
# ============================================================

def demo_advanced():
    """
    silent_errors=True 跳过无法加载的文件（如编码错误），不中断整个流程。
    """
    print("\n=== 演示 3：高级用法 ===")

    # 静默跳过加载失败的文件
    loader = DirectoryLoader(
        "./docs",
        glob="**/*.txt",
        loader_cls=TextLoader,
        silent_errors=True,    # 跳过加载失败的文件
        show_progress=True,    # 显示进度条
    )
    docs = loader.load()
    print(f"成功加载 {len(docs)} 个文档")

    # 懒加载（逐文件处理，适合大量文件）
    print("\n懒加载模式:")
    loader_lazy = DirectoryLoader(
        "./docs",
        glob="**/*.txt",
        loader_cls=TextLoader,
    )
    for i, doc in enumerate(loader_lazy.lazy_load()):
        print(f"  [{i}] {os.path.basename(doc.metadata['source'])}: {len(doc.page_content)} 字符")


# ============================================================
# 演示 4：自定义 Loader
# ============================================================

def demo_custom_loader():
    """
    DirectoryLoader 的 loader_cls 可以是任何 LangChain Loader。
    例如：CSVLoader、UnstructuredHTMLLoader、PyPDFLoader 等。
    """
    print("\n=== 演示 4：自定义 Loader ===")

    print("常用组合:")
    print("  loader_cls=TextLoader        -> 纯文本文件")
    print("  loader_cls=CSVLoader         -> CSV 文件")
    print("  loader_cls=PyPDFLoader       -> PDF 文件")
    print("  loader_cls=UnstructuredHTMLLoader -> HTML 文件")
    print("  loader_cls=PythonLoader      -> Python 源码文件")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    docs_dir = "docs"
    os.makedirs(os.path.join(docs_dir, "sub"), exist_ok=True)

    files = {
        "docs/intro.txt": "LangChain 简介\n这是一个介绍文件。",
        "docs/tutorial.txt": "LangChain 教程\n这是教程文件，内容更多一些。",
        "docs/sub/notes.txt": "学习笔记\n这是子目录中的笔记文件。",
    }
    for path, content in files.items():
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    demo_basic()
    demo_glob_patterns()
    demo_advanced()
    demo_custom_loader()

    shutil.rmtree(docs_dir, ignore_errors=True)
