"""
05_tools_summary.py - Tools 模块总结

汇总 Tools 模块的核心知识点，提供一份速查表。

本模块覆盖：
  01_tool_decorator.py  — @tool 装饰器定义工具
  02_structured_tool.py — StructuredTool 显式创建工具
  03_tool_calling.py    — bind_tools + tool calling 流程

参考文档：
  - Tools 概述: https://docs.langchain.com/oss/python/langchain/tools
  - Tool Calling: https://docs.langchain.com/oss/python/langchain/models#tool-calling
"""

from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field


# ============================================================
# 1. 工具定义方式速查
# ============================================================

# 方式 A：@tool 装饰器（推荐，90% 场景够用）
@tool
def quick_tool(query: str) -> str:
    """一句话描述工具功能"""
    return f"结果: {query}"


# 方式 B：@tool + Pydantic（参数多、需要详细描述时）
class SearchInput(BaseModel):
    keyword: str = Field(description="搜索关键词")
    limit: int = Field(default=10, description="返回数量上限")

@tool(args_schema=SearchInput)
def search_with_schema(keyword: str, limit: int = 10) -> str:
    """带详细参数描述的搜索工具"""
    return f"搜索 '{keyword}'，限制 {limit} 条"


# 方式 C：StructuredTool（动态生成、包装已有函数）
def existing_func(x: int) -> int:
    return x * 2

wrapped_tool = StructuredTool.from_function(
    func=existing_func,
    name="double",
    description="将输入数字翻倍",
)


# ============================================================
# 2. Tool Calling 流程速查
# ============================================================

TOOL_CALLING_FLOW = """
Tool Calling 完整流程：

  用户提问
    ↓
  model.bind_tools(tools).invoke(question)
    ↓
  AIMessage(tool_calls=[{name, args, id}])   ← 模型决定调哪个工具
    ↓
  执行工具: tool_map[name].invoke(args)
    ↓
  ToolMessage(content=结果, tool_call_id=id)  ← 把结果传回模型
    ↓
  model.invoke([HumanMsg, AIMsg, ToolMsg])    ← 模型生成最终回复

要点：
  - bind_tools() 把工具 schema 注册给模型
  - tool_calls 中的 id 必须和 ToolMessage 中的 tool_call_id 对应
  - 一次可以返回多个 tool_calls（并行调用）
  - 模型可能需要多轮工具调用（串行调用）
"""


# ============================================================
# 3. 核心对象速查
# ============================================================

CORE_OBJECTS = """
核心对象：

  @tool 装饰器
    输入: 函数（带类型注解和 docstring）
    输出: StructuredTool 实例
    关键属性: .name, .description, .args, .args_schema

  StructuredTool
    创建方式: from_function(func, name, description, args_schema)
    关键属性: .name, .description, .args_schema, .invoke()

  AIMessage.tool_calls
    类型: list[dict]
    结构: [{"name": str, "args": dict, "id": str, "type": "tool_call"}]

  ToolMessage
    创建: ToolMessage(content=str, tool_call_id=str)
    作用: 把工具执行结果传回模型

  bind_tools(tools, tool_choice=...)
    tool_choice: "auto"(默认) | "any"(必须调) | "none"(禁止) | {"type":"tool","name":"xxx"}
"""


# ============================================================
# 4. 常见陷阱
# ============================================================

COMMON_PITFALLS = """
常见陷阱：

  ❌ docstring 写得太模糊 → LLM 不知道什么时候该调这个工具
  ✅ docstring 要写清楚：什么情况下用、输入是什么、返回什么

  ❌ 忘记加类型注解 → args_schema 无法正确生成
  ✅ 每个参数都要有类型注解，可选参数用 Optional 或默认值

  ❌ ToolMessage 的 tool_call_id 和 tool_calls 的 id 不匹配
  ✅ 必须一一对应，否则模型无法关联

  ❌ 工具返回复杂对象（dict/list） → LLM 收到的可能是 str(repr(...))
  ✅ 工具返回值推荐用 str，复杂结构用 json.dumps()

  ❌ 没有对工具执行做超时/错误处理
  ✅ 用 try-except 包裹，把错误信息通过 ToolMessage 告诉模型
"""


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LangChain Tools 模块总结")
    print("=" * 60)

    print("\n--- Tool Calling 流程 ---")
    print(TOOL_CALLING_FLOW)

    print("--- 核心对象 ---")
    print(CORE_OBJECTS)

    print("--- 常见陷阱 ---")
    print(COMMON_PITFALLS)
