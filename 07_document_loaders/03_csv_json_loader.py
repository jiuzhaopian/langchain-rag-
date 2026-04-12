"""
03_csv_json_loader.py - CSV 和 JSON 加载器

除纯文本外，LangChain 还提供了结构化数据的加载器：
- CSVLoader: 将 CSV 文件每行加载为一个 Document
- JSONLoader: 用 jq 表达式从 JSON 中提取字段作为 Document 内容

参考文档：
  - CSVLoader: https://python.langchain.com/docs/integrations/document_loaders/csv/
  - JSONLoader: https://python.langchain.com/docs/integrations/document_loaders/json/

安装：
  pip install langchain-community
"""

import json
import os
import csv


# ============================================================
# 公共：创建示例数据文件
# ============================================================

def create_sample_files():
    """创建 CSV 和 JSON 示例文件"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # CSV 文件
    csv_path = os.path.join(script_dir, "sample.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "description", "category"])
            writer.writerow(["LangChain", "LLM 应用开发框架", "AI"])
            writer.writerow(["LlamaIndex", "数据索引框架", "AI"])
            writer.writerow(["FastAPI", "Python Web 框架", "Web"])

    # JSON 文件
    json_path = os.path.join(script_dir, "sample.json")
    if not os.path.exists(json_path):
        data = [
            {"name": "LangChain", "desc": "LLM 应用开发框架", "category": "AI"},
            {"name": "LlamaIndex", "desc": "数据索引框架", "category": "AI"},
            {"name": "FastAPI", "desc": "Python Web 框架", "category": "Web"},
        ]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # 嵌套 JSON 文件
    nested_json_path = os.path.join(script_dir, "nested_sample.json")
    if not os.path.exists(nested_json_path):
        data = {
            "frameworks": [
                {"name": "LangChain", "details": {"category": "AI", "lang": "Python"}},
                {"name": "Django", "details": {"category": "Web", "lang": "Python"}},
            ]
        }
        with open(nested_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return csv_path, json_path, nested_json_path


# ============================================================
# 演示 1：CSVLoader 基本用法
# ============================================================

def demo_csv_loader(csv_path):
    """
    CSVLoader 将 CSV 每行加载为一个 Document。
    page_content = 该行所有列拼接（默认用逗号分隔）。
    """
    print("=== 演示 1：CSVLoader 基本用法 ===")

    from langchain_community.document_loaders import CSVLoader

    loader = CSVLoader(csv_path)
    docs = loader.load()

    print(f"加载了 {len(docs)} 个文档（CSV 文件有 3 行数据 + 1 行表头）")
    for doc in docs:
        print(f"  行号 {doc.metadata.get('row', '?')}: {doc.page_content[:80]}")

    print(f"\n元数据字段: {list(docs[0].metadata.keys())}")


# ============================================================
# 演示 2：CSVLoader 高级用法
# ============================================================

def demo_csv_advanced(csv_path):
    """
    CSVLoader 支持指定列、自定义分隔符等。
    """
    print("\n=== 演示 2：CSVLoader 高级用法 ===")

    from langchain_community.document_loaders import CSVLoader

    # source_column: 将某一列的值作为 metadata.source
    loader = CSVLoader(csv_path, source_column="name")
    docs = loader.load()
    for doc in docs:
        print(f"  source={doc.metadata.get('source')}: {doc.page_content[:60]}")

    # csv_args: 传入 csv 模块参数（如指定分隔符）
    loader2 = CSVLoader(
        csv_path,
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
        },
    )
    print(f"\ncsv_args 自定义: 加载 {len(loader2.load())} 个文档")


# ============================================================
# 演示 3：JSONLoader 基本用法
# ============================================================

def demo_json_loader(json_path):
    """
    JSONLoader 用 jq 表达式提取 JSON 中的内容。
    需要安装 jq: pip install jq
    """
    print("\n=== 演示 3：JSONLoader 基本用法 ===")

    try:
        from langchain_community.document_loaders import JSONLoader

        # jq 表达式 ".[]" 遍历数组每个元素
        loader = JSONLoader(
            json_path,
            jq_schema=".[]",
            text_content=False,  # False: 提取到的对象转为 JSON 字符串
        )
    except ImportError as e:
        print(f"跳过: 需要安装 jq（{e}）")
        print("  安装: pip install jq")
        return

    docs = loader.load()
    print(f"jq='.[]'（遍历数组）: {len(docs)} 个文档")
    for doc in docs:
        print(f"  {doc.page_content[:80]}")

    # 只提取 name 字段
    loader2 = JSONLoader(
        json_path,
        jq_schema=".[].name",
        text_content=True,  # True: 直接用字符串值（不包 JSON）
    )
    docs2 = loader2.load()
    print(f"\njq='.[].name'（提取 name 字段）: {len(docs2)} 个文档")
    for doc in docs2:
        print(f"  {doc.page_content}")

    # 提取多个字段拼接
    loader3 = JSONLoader(
        json_path,
        jq_schema='.[] | "工具: " + .name + " - " + .desc',
        text_content=True,
    )
    docs3 = loader3.load()
    print(f"\njq 拼接多个字段: {len(docs3)} 个文档")
    for doc in docs3:
        print(f"  {doc.page_content}")


# ============================================================
# 演示 4：JSONLoader 嵌套 JSON
# ============================================================

def demo_json_nested(nested_json_path):
    """
    处理嵌套 JSON 结构。
    """
    print("\n=== 演示 4：嵌套 JSON ===")

    try:
        from langchain_community.document_loaders import JSONLoader
        loader = JSONLoader(
            nested_json_path,
            jq_schema=".frameworks[]",
            text_content=False,
        )
    except ImportError:
        print("跳过: 需要 pip install jq")
        return
    docs = loader.load()
    print(f"jq='.frameworks[]': {len(docs)} 个文档")
    for doc in docs:
        data = json.loads(doc.page_content)
        print(f"  {data['name']}: {data['details']}")


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    csv_path, json_path, nested_json_path = create_sample_files()
    demo_csv_loader(csv_path)
    demo_csv_advanced(csv_path)
    demo_json_loader(json_path)
    demo_json_nested(nested_json_path)
