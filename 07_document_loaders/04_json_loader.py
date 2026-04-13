"""
04_json_loader.py - JSON 加载器

JSONLoader 用 jq 表达式从 JSON 文件中提取内容，生成 Document 列表。
每个匹配结果对应一个 Document。

参考文档：
  - JSONLoader: https://docs.langchain.com/oss/python/integrations/document_loaders/json
  - jq 语法教程: https://jqlang.github.io/jq/manual/

安装：
  pip install langchain-community jq
  # jq 是可选依赖，未安装时 JSONLoader 会抛出 ImportError

jq 常用语法速查（配合 JSONLoader 的 jq_schema 参数）：
  基础选择：
    .          当前对象（整个输入）
    .key       取对象的字段值（如 .name 取 name 字段）
    .[index]   取数组元素（从 0 开始，如 .[0] 取第一个）
    .[]        遍历数组所有元素，逐个输出
  管道组合：
    .[] | .key      遍历数组，取每个元素的 key 字段
    .[] | .a + .b   遍历数组，字符串拼接两个字段（无分隔符）
  过滤筛选：
    .[] | select(.status == "active")   只保留 status 为 active 的元素
    .[] | select(.age > 18)              只保留 age 大于 18 的元素
  构造输出：
    .[] | tostring                        将对象转为 JSON 字符串（保留中文）
    .[] | "名称: " + .name               拼接固定文本和字段值
    .[] | {name, desc}                   只保留指定字段，构造新对象
  嵌套访问：
    .frameworks[]           访问嵌套数组
    .users[0].address.city  多层嵌套取值
"""

import json
import os

from langchain_community.document_loaders import JSONLoader  # noqa: 构造函数需要 jq 可选依赖


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ============================================================
# 公共：创建示例数据文件
# ============================================================

def create_sample_json():
    """创建 JSON 示例文件"""
    os.makedirs(DATA_DIR, exist_ok=True)

    json_path = os.path.join(DATA_DIR, "sample.json")
    if not os.path.exists(json_path):
        data = [
            {"name": "LangChain", "desc": "LLM 应用开发框架", "category": "AI"},
            {"name": "LlamaIndex", "desc": "数据索引框架", "category": "AI"},
            {"name": "FastAPI", "desc": "Python Web 框架", "category": "Web"},
        ]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    nested_json_path = os.path.join(DATA_DIR, "nested_sample.json")
    if not os.path.exists(nested_json_path):
        data = {
            "frameworks": [
                {"name": "LangChain", "details": {"category": "AI", "lang": "Python"}},
                {"name": "Django", "details": {"category": "Web", "lang": "Python"}},
            ]
        }
        with open(nested_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return json_path, nested_json_path


# ============================================================
# 演示 1：JSONLoader 基本用法
# ============================================================

def demo_json_basic(json_path):
    """
    JSONLoader 通过 jq 表达式定位要提取的 JSON 片段。
    jq_schema 定义提取规则，每个匹配结果生成一个 Document。

    text_content 参数：
      - False（默认）: jq 输出再经 json.dumps() 序列化后存入 page_content。
        json.dumps() 默认 ensure_ascii=True，中文被转义为 \\uXXXX。
      - True: jq 输出直接作为 page_content，不再经过 json.dumps()，中文保持原样。
    """
    print("=== 演示 1：JSONLoader 基本用法 ===")

    try:
        # jq=".[]" 遍历数组每个元素
        loader = JSONLoader(
            json_path,
            jq_schema=".[]",
            text_content=False,
        )
    except ImportError:
        print("跳过: 需要安装 jq（pip install jq）")
        return
    docs = loader.load()
    print(f"jq='.[]'（遍历数组，text_content=False）: {len(docs)} 个文档")
    print(f"  注意：中文被 json.dumps 转义为 \\\\uXXXX")
    for doc in docs:
        print(f"  {doc.page_content[:80]}")

    # text_content=True 提取整个对象（避免中文转义）
    # 通过 jq 的 tostring 内置函数将对象转为字符串，此时中文正常显示
    loader0 = JSONLoader(
        json_path,
        jq_schema='.[] | tostring',
        text_content=True,
    )
    docs0 = loader0.load()
    print(f"\njq='.[] | tostring'（遍历数组，text_content=True）: {len(docs0)} 个文档")
    for doc in docs0:
        print(f"  {doc.page_content[:80]}")

    # 只提取 name 字段
    loader2 = JSONLoader(
        json_path,
        jq_schema=".[].name",
        text_content=True,  # True: 直接用字符串值
    )
    docs2 = loader2.load()
    print(f"\njq='.[].name'（提取单个字段）: {len(docs2)} 个文档")
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
# 演示 2：嵌套 JSON
# ============================================================

def demo_json_nested(nested_json_path):
    """
    处理嵌套 JSON 结构：用 jq 指定路径定位到目标数组。
    """
    print("\n=== 演示 2：嵌套 JSON ===")

    try:
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
    json_path, nested_json_path = create_sample_json()
    demo_json_basic(json_path)
    demo_json_nested(nested_json_path)
