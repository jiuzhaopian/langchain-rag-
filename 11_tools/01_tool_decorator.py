"""
01_tool_decorator.py
LangChain Tools - @tool 装饰器定义工具

工具（Tool）是 Agent 调用外部功能的基本单元。
@tool 是最简单的工具定义方式：写一个函数，加装饰器就行。

核心概念：
  - 函数签名（参数名 + 类型注解）= 工具的输入 schema
  - 函数的 docstring = 工具的描述（LLM 靠这个决定什么时候调用）
  - @tool 会自动把函数转成 StructuredTool 对象

参考文档：
  - Tools 概述: https://docs.langchain.com/oss/python/langchain/tools
  - StructuredTool API: https://reference.langchain.com/python/langchain-core/tools/structured/StructuredTool

安装：
  pip install langchain-core
"""

from langchain_core.tools import tool


# ============================================================
# 演示 1：最基本的工具定义
# ============================================================

def demo_basic_tool():
    """
    @tool 装饰器自动从函数提取三个信息：
      1. 函数名 → 工具名
      2. docstring → 工具描述（LLM 靠描述决定何时调用）
      3. 参数类型注解 → 工具的输入 schema

    另外@tool装饰器本身可以传许多参数(name,description等),且部分参数优先级更高

    注意：docstring 非常重要，描述写得越清楚，LLM 越能正确选择工具。
    """
    @tool
    def get_word_length(word: str) -> int:
        """计算一个单词的字符长度"""
        return len(word)

    # 查看工具信息
    print("=== 基本工具 ===")
    print(f"工具名: {get_word_length.name}")
    print(f"工具描述: {get_word_length.description}")
    print(f"参数 schema: {get_word_length.args_schema.model_json_schema()}")
    print()

    # 直接调用（和普通函数一样）
    result = get_word_length.invoke("langchain")
    print(f"get_word_length('langchain') = {result}")
    print(f"返回类型: {type(result).__name__}")
    print()


# ============================================================
# 演示 2：多个参数的工具
# ============================================================

def demo_multi_param_tool():
    """
    多个参数时，每个参数都需要类型注解。
    可选参数用默认值表示。
    """
    @tool
    def calculate_bmi(weight_kg: float, height_m: float) -> str:
        """根据体重(kg)和身高(m)计算BMI指数，返回分类结果"""
        bmi = weight_kg / (height_m ** 2)
        if bmi < 18.5:
            category = "偏瘦"
        elif bmi < 24:
            category = "正常"
        elif bmi < 28:
            category = "偏胖"
        else:
            category = "肥胖"
        return f"BMI={bmi:.1f}，属于{category}"

    print("=== 多参数工具 ===")
    print(f"工具名: {calculate_bmi.name}")
    print(f"参数: {calculate_bmi.args_schema.model_json_schema()}")
    result = calculate_bmi.invoke({"weight_kg": 70, "height_m": 1.75})
    print(f"结果: {result}")
    print()


# ============================================================
# 演示 3：返回类型的影响
# ============================================================

def demo_return_types():
    """
    工具的返回值会被自动转成字符串给 LLM。
    返回值类型的不同不影响工具本身，但影响 LLM 收到的信息。

    最佳实践：
      - 简单工具：直接返回 str
      - 数值工具：返回数值，LCEL 会自动处理
      - 复杂工具：返回 JSON 字符串，方便 LLM 解析
    """
    @tool
    def multiply(a: int, b: int) -> int:
        """计算两个整数的乘积"""
        return a * b

    @tool
    def search_city_info(city: str) -> str:
        """查询城市基本信息（模拟）"""
        # 模拟数据，实际场景中会调 API 或查数据库
        data = {
            "北京": '{"name":"北京","population":"2189万","area":"16410km²"}',
            "上海": '{"name":"上海","population":"2487万","area":"6341km²"}',
        }
        return data.get(city, f'未找到{city}的信息')

    print("=== 返回类型 ===")
    print(f"multiply(3, 4) = {multiply.invoke({'a': 3, 'b': 4})}")
    print(f"search_city_info('北京') = {search_city_info.invoke({'city': '北京'})}")
    print()


# ============================================================
# 演示 4：可选参数和默认值
# ============================================================

def demo_optional_params():
    """
    参数可以有默认值，LLM 可以选择不传这些参数。
    """
    from typing import Optional

    @tool
    def format_text(text: str, max_length: Optional[int] = None, prefix: str = "") -> str:
        """格式化文本，可限制最大长度和添加前缀"""
        result = prefix + text
        if max_length and len(result) > max_length:
            result = result[:max_length] + "..."
        return result

    print("=== 可选参数 ===")
    schema = format_text.args_schema.model_json_schema()
    print(f"参数 schema: {schema}")
    # 只传必填参数
    print(f"format_text('hello') = {format_text.invoke({'text': 'hello'})}")
    # 传所有参数
    print(f"format_text('hello world', max_length=8, prefix='>> ') = "
          f"{format_text.invoke({'text': 'hello world', 'max_length': 8, 'prefix': '>> '})}")
    print()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    demo_basic_tool()
    demo_multi_param_tool()
    demo_return_types()
    demo_optional_params()
