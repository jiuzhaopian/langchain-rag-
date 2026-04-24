"""
03_csv_loader.py - CSV 加载器

CSVLoader 将 CSV 文件每行加载为一个 Document，page_content 为该行所有列拼接。

参考文档：
  - CSVLoader: https://docs.langchain.com/oss/python/integrations/document_loaders/csv

安装：
  pip install langchain-community
"""

import csv
import os

from langchain_community.document_loaders import CSVLoader

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ============================================================
# 公共：创建示例数据文件
# ============================================================

def create_sample_csv():
    """创建 CSV 示例文件"""
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "sample.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "description", "category"])
            writer.writerow(["LangChain", "LLM 应用开发框架", "AI"])
            writer.writerow(["LlamaIndex", "数据索引框架", "AI"])
            writer.writerow(["FastAPI", "Python Web 框架", "Web"])
    return csv_path


def create_no_header_csv():
    """创建无表头 CSV 示例文件"""
    os.makedirs(DATA_DIR, exist_ok=True)
    no_header_path = os.path.join(DATA_DIR, "sample_no_header.csv")
    if not os.path.exists(no_header_path):
        with open(no_header_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["LangChain", "LLM 应用开发框架", "AI"])
            writer.writerow(["FastAPI", "Python Web 框架", "Web"])
    return no_header_path


# ============================================================
# 演示 1：CSVLoader 基本用法
# ============================================================

def demo_csv_basic(csv_path):
    """
    CSVLoader 将 CSV 每行加载为一个 Document。
    page_content 格式：每列一行，"列名: 值"，多列用换行符拼接。
    """
    print("=== 演示 1：CSVLoader 基本用法 ===")
    # TODO
    pass


# ============================================================
# 演示 2：CSVLoader 高级用法
# ============================================================

def demo_csv_advanced(csv_path):
    """
    CSVLoader 支持指定列、自定义分隔符等高级参数。
    """
    print("\n=== 演示 2：CSVLoader 高级用法 ===")

    # source_column: 将某一列的值作为 metadata.source（默认是文件路径）
    # metadata_columns:指定哪些列从 page_content 中移除，改放到 metadata 里
    # content_columns: 指定只有这些列进 page_content，其余列全部丢弃（既不进 content 也不进 metadata）

    loader = CSVLoader(csv_path, source_column="name")
    docs = loader.load()
    for doc in docs:
        print(f"  【metadata】row {doc.metadata.get('row', '?')}; source: {doc.metadata.get('source', 'x')}")
        print(f"  【内容】:\n {doc.page_content}")

    # csv_args: 传入 Python csv 模块的参数，控制 CSV 解析行为
    # 常用参数：
    #   delimiter  - 字段分隔符，默认逗号 ","。改为 "\t" 可解析 TSV 文件
    #   quotechar  - 引号字符，默认双引号 '"'。用于包裹含分隔符的字段
    #   fieldnames - 字段名列表。如果 CSV 文件没有表头行，必须手动指定
    #   skipinitialspace - 是否跳过分隔符后的空格，默认 False
    # 更多参数见: https://docs.python.org/3/library/csv.html
    loader2 = CSVLoader(
        csv_path,
        csv_args={
            "delimiter": ",",   # 字段分隔符
            "quotechar": '"',   # 引号字符
        },
    )
    print(f"\ncsv_args 自定义: 加载 {len(loader2.load())} 个文档")


# ============================================================
# 演示 3：无表头 CSV
# ============================================================

def demo_csv_no_header(no_header_path):
    """
    CSV 文件没有表头时，需要通过 csv_args 指定 fieldnames。
    """
    print("\n=== 演示 3：无表头 CSV ===")

    # 不指定 fieldnames，第一行数据会被当作表头丢失
    loader_no_names = CSVLoader(no_header_path)
    docs_no_names = loader_no_names.load()
    print(f"不指定 fieldnames: {len(docs_no_names)} 个文档")
    for doc in docs_no_names:
        print(f"  {doc.page_content}")

    # 指定 fieldnames，所有行都会被加载为 Document
    loader_with_names = CSVLoader(
        no_header_path,
        csv_args={"fieldnames": ["name", "description", "category"]},
    )
    docs_with_names = loader_with_names.load()
    print(f"\n指定 fieldnames: {len(docs_with_names)} 个文档")
    for doc in docs_with_names:
        print(f"  {doc.page_content}")


if __name__ == "__main__":
    csv_path = create_sample_csv()
    no_header_path = create_no_header_csv()
    # 演示
    demo_csv_basic(csv_path)
    demo_csv_advanced(csv_path)
    demo_csv_no_header(no_header_path)
