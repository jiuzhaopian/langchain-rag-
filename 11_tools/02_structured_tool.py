"""
02_structured_tool.py
LangChain Tools - StructuredTool 显式创建工具

当 @tool 装饰器不够灵活时（比如需要动态生成工具、工具逻辑不在函数里、第三方api、需要自定义描述等），可以用 StructuredTool 显式创建。

核心区别：
  - @tool: 自动从函数提取信息，简单快捷
  - StructuredTool: 手动指定一切，灵活可控

参考文档：
  - StructuredTool API: https://reference.langchain.com/python/langchain-core/tools/structured/StructuredTool
  - Custom Tools: https://docs.langchain.com/oss/python/langchain/tools

安装：
  pip install langchain-core
"""

from langchain_core.tools import StructuredTool, tool
from pydantic import BaseModel, Field


# ============================================================
# 演示 1：StructuredTool 基本用法
# ============================================================

def demo_basic_structured_tool():
    """
    StructuredTool 需要手动提供：name、description、func、args_schema
    适用于：工具逻辑已存在（第三方库函数），不方便加 @tool 装饰器。
    """
    def get_weather(city: str) -> str:
        # 模拟天气 API
        return f"{city}今天晴，温度25°C"

    weather_tool = StructuredTool.from_function(
        func=get_weather,
        name="get_weather",
        description="查询指定城市的天气状况"
    )

    print("=== StructuredTool 基本用法 ===")
    print(f"工具名: {weather_tool.name}")
    print(f"工具描述: {weather_tool.description}")
    print(f"结果: {weather_tool.invoke({'city': '北京'})}")
    print()


# ============================================================
# 演示 2：用 Pydantic 定义参数 schema
# ============================================================

def demo_pydantic_schema():
    """
    用 Pydantic BaseModel 定义参数，可以加 Field 描述。
    Field 描述会出现在 JSON Schema 中，帮助 LLM 更好地理解参数含义。

    这是推荐的做法：参数少时 @tool 就够了，参数多或需要详细描述时用 Pydantic。
    """

    class SearchInput(BaseModel):
        query: str = Field(description="搜索关键词")
        max_results: int = Field(default=5, description="最多返回结果数")
        language: str = Field(default="zh", description="结果语言，如 zh/en")

    def search_func(query: str, max_results: int = 5, language: str = "zh") -> str:
        # 模拟搜索
        return f"搜索 '{query}'，返回 {max_results} 条 {language} 结果"

    search_tool = StructuredTool.from_function(
        func=search_func,
        name="web_search",
        description="在网络上搜索信息",
        args_schema=SearchInput,
    )

    print("=== Pydantic 参数 schema ===")
    schema = search_tool.args_schema.model_json_schema()
    print(f"JSON Schema:\n{schema}")
    print()

    # 调用
    result = search_tool.invoke({"query": "LangChain 教程"})
    print(f"结果: {result}")
    print()


# ============================================================
# 演示 3：@tool 装饰器 + Pydantic schema（推荐写法）
# ============================================================

def demo_tool_with_pydantic():
    """
    @tool 也可以配合 Pydantic schema 使用：
    把 Pydantic model 作为唯一参数类型，@tool 会自动识别。

    这是实际项目中最常用的写法：既有 @tool 的简洁，又有 Pydantic 的描述能力。
    """

    class CalculatorInput(BaseModel):
        a: float = Field(description="第一个数")
        b: float = Field(description="第二个数")
        operation: str = Field(description="运算类型：add/sub/mul/div")
        desc: str | None= Field(description="说明",default=None)

    @tool(args_schema=CalculatorInput)
    def calculator(a: float, b: float, operation: str,desc:str|None) -> str:
        """执行基本数学运算"""
        ops = {
            "add": lambda x, y: x + y,
            "sub": lambda x, y: x - y,
            "mul": lambda x, y: x * y,
            "div": lambda x, y: x / y if y != 0 else "错误：除数不能为零",
        }
        if operation not in ops:
            return f"不支持的运算类型：{operation}，支持：add/sub/mul/div"
        result = ops[operation](a, b)
        return f"{a} {operation} {b} = {result}"

    print("=== @tool + Pydantic（推荐） ===")
    print(f"工具描述: {calculator.description}")
    schema = calculator.args_schema.model_json_schema()
    print(f"参数 schema:\n{schema}")
    print()

    # 测试
    print(f"calculator(10, 3, 'div') = {calculator.invoke({'a': 10, 'b': 3, 'operation': 'div'})}")
    print(f"calculator(10, 3, 'add') = {calculator.invoke({'a': 10, 'b': 3, 'operation': 'add'})}")
    print()


# ============================================================
# 演示 4：@tool vs StructuredTool 对比
# ============================================================

def demo_comparison():
    """
    两种方式创建的工具，效果完全一样。
    选择依据：
      - @tool: 90% 场景够用，简洁
      - StructuredTool: 需要动态生成、包装已有函数、复杂 schema 时
    """
    # 方式 1：@tool
    @tool
    def add_tool(a: int, b: int) -> int:
        """两数相加"""
        return a + b

    # 方式 2：StructuredTool
    def sub_func(a: int, b: int) -> int:
        return a - b

    sub_tool = StructuredTool.from_function(
        func=sub_func,
        name="subtract",
        description="两数相减",
    )

    print("=== @tool vs StructuredTool ===")
    print(f"@tool 类型: {type(add_tool).__name__}")
    print(f"StructuredTool 类型: {type(sub_tool).__name__}")
    print(f"add(3, 2) = {add_tool.invoke({'a': 3, 'b': 2})}")
    print(f"sub(3, 2) = {sub_tool.invoke({'a': 3, 'b': 2})}")
    # 两者都是 StructuredTool 实例，行为一致
    print(f"类型相同: {type(add_tool) == type(sub_tool)}")
    print()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    demo_basic_structured_tool()
    demo_pydantic_schema()
    demo_tool_with_pydantic()
    demo_comparison()
