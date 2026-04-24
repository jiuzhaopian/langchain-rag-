"""
02_directory_loader.py - DirectoryLoader 目录加载器

DirectoryLoader 扫描目录，将匹配的文件批量加载为 Document 列表。
本质是遍历文件 + 对每个文件调用指定的 Loader。

重要：loader_cls 默认是 UnstructuredFileLoader（需要安装 unstructured 包），
       通常需要手动指定为 TextLoader。

参考文档：
  - DirectoryLoader: https://reference.langchain.com/python/langchain-community/document_loaders/directory/DirectoryLoader

安装：
  pip install langchain-community
"""

import os
import shutil

from langchain_community.document_loaders import DirectoryLoader, TextLoader

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TEST_DIR = os.path.join(DATA_DIR, "docs_test")


# ============================================================
# 公共：创建测试目录和文件
# ============================================================

def create_test_files():
    """创建测试用的目录结构"""
    os.makedirs(os.path.join(TEST_DIR, "sub"), exist_ok=True)

    files = {
        os.path.join(TEST_DIR, "intro.txt"): "LangChain 简介\n这是一个介绍文件。",
        os.path.join(TEST_DIR, "tutorial.txt"): "LangChain 教程\n这是教程文件，内容更多一些。",
        os.path.join(TEST_DIR, "sub", "notes.txt"): "学习笔记\n这是子目录中的笔记文件。",
    }
    for path, content in files.items():
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


# ============================================================
# 演示 1：基本用法
# ============================================================

def demo_basic():
    """
    扫描目录，加载所有匹配的文件。
    loader_cls 必须指定，否则默认 UnstructuredFileLoader（需安装 unstructured）。
    """
    print("=== 演示 1：基本用法 ===")
    # TODO
    pass


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
# 演示 3：高级用法（silent_errors + 懒加载）
# ============================================================

def demo_advanced():
    """
    silent_errors=True 跳过无法加载的文件，不中断整个流程。
    lazy_load() 逐文件处理，适合大量文件场景。
    """
    print("\n=== 演示 3：高级用法 ===")

    # 静默跳过加载失败的文件
    loader = DirectoryLoader(
        TEST_DIR,
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
        TEST_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
    )
    for i, doc in enumerate(loader_lazy.lazy_load()):
        print(f"  [{i}] {os.path.basename(doc.metadata['source'])}: {len(doc.page_content)} 字符")


if __name__ == "__main__":
    create_test_files()
    demo_basic()
    demo_glob_patterns()
    demo_advanced()

    # 清理测试目录
    shutil.rmtree(TEST_DIR, ignore_errors=True)
