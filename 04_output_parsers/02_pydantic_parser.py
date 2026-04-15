"""02_pydantic_parser.py - PydanticOutputParser (结构化解析)

PydanticOutputParser 是最实用的 Output Parser：
  - 用 Pydantic 模型定义输出结构
  - 自动生成格式指令注入 prompt
  - invoke 返回 Pydantic 对象（不是 dict）
  - 自动验证字段类型（失败会报 OutputParserException）

对比 JsonOutputParser：
  - JsonOutputParser 返回 dict，不做类型验证
  - PydanticOutputParser 返回 Pydantic 对象，做严格类型验证

Pydantic 基础知识：
  - BaseModel: Pydantic 的数据模型基类，继承它即可定义结构化数据
    提供自动类型转换、数据验证、序列化等能力
  - Field(): 为字段添加元数据，如 description（描述字段含义，用于生成格式指令）
    LLM 看到的格式指令中，字段说明就来自 Field(description=...)
  - 类型注解（str, int, list[str] 等）：声明字段的期望类型，Pydantic 会自动验证

参考文档：
  - PydanticOutputParser API: https://reference.langchain.com/python/langchain-core/output_parsers/pydantic/PydanticOutputParser
  - Pydantic BaseModel 文档: https://docs.pydantic.dev/latest/concepts/models/
  - 源码: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/output_parsers/pydantic.py

安装：
  pip install langchain-core langchain-community zhipuai pydantic
"""

import json
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate


# ========================= 模型配置 =========================

def get_llm():
    from langchain_community.chat_models import ChatZhipuAI
    import os
    return ChatZhipuAI(
        model="glm-4.7",
        api_key=os.environ.get("ZHIPUAI_API_KEY"),
    )


# ============================================================
# 演示 1：基本用法
# ============================================================

def demo_basic():
    """
    步骤：
    1. 定义 Pydantic 模型（描述输出结构）
       - 继承 BaseModel，这是 Pydantic 数据模型的基类
       - 每个字段用类型注解声明类型（str, int 等）
       - 用 Field(description=...) 描述字段含义，parser 会用它生成格式指令
    2. 创建 PydanticOutputParser(pydantic_object=你的模型)
    3. 用 parser.get_format_instructions() 注入 prompt
    4. 链式调用：prompt | llm | parser
    5. 返回的是 Pydantic 对象，用 .字段名 访问
    """
    class BookInfo(BaseModel):
        title: str = Field(description="书名")
        author: str = Field(description="作者")
        year: int = Field(description="出版年份")
        genre: str = Field(description="类型，如科幻/文学/历史等")

    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=BookInfo)

    # 查看自动生成的格式指令
    print("=== PydanticOutputParser 格式指令 ===")
    print(parser.get_format_instructions())
    print()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个图书信息助手。{format_instructions}"),
        ("human", "提供《三体》的信息"),
    ])

    chain = prompt | llm | parser
    result = chain.invoke({
        "format_instructions": parser.get_format_instructions(),
    })

    print("=== 解析结果 ===")
    print(f"类型: {type(result).__name__}")
    print(f"是 BookInfo 实例: {isinstance(result, BookInfo)}")
    print(f"title: {result.title}")
    print(f"author: {result.author}")
    print(f"year: {result.year}")
    print(f"genre: {result.genre}")
    print()


# ============================================================
# 演示 2：嵌套结构
# ============================================================

def demo_nested():
    """
    Pydantic 模型中可以嵌套其他 BaseModel，构建复杂结构。
    """
    class Address(BaseModel):
        city: str = Field(description="城市")
        district: str = Field(description="区/县")

    class PersonInfo(BaseModel):
        name: str = Field(description="姓名")
        age: int = Field(description="年龄")
        address: Address = Field(description="地址")

    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=PersonInfo)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个信息提取助手。{format_instructions}"),
        ("human", "提取：李明，30岁，住在北京市海淀区"),
    ])

    chain = prompt | llm | parser
    result = chain.invoke({
        "format_instructions": parser.get_format_instructions(),
    })

    print("=== 嵌套结构 ===")
    print(f"name: {result.name}, age: {result.age}, address: {result.address.city}, {result.address.district}")
    print()


# ============================================================
# 演示 3：列表字段
# ============================================================

def demo_list_field():
    """
    Pydantic 支持列表类型字段，让 LLM 返回列表数据。
    """
    class Recipe(BaseModel):
        name: str = Field(description="菜名")
        ingredients: list[str] = Field(description="食材列表")
        steps: list[str] = Field(description="步骤列表")
        difficulty: str = Field(description="难度：简单/中等/困难")

    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=Recipe)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个菜谱助手。{format_instructions}"),
        ("human", "提供番茄炒蛋的做法"),
    ])

    chain = prompt | llm | parser
    result = chain.invoke({
        "format_instructions": parser.get_format_instructions(),
    })

    print("=== 列表字段 ===")
    print(f"菜名: {result.name}")
    print(f"食材: {result.ingredients}")
    print(f"步骤数: {len(result.steps)}")
    print(f"难度: {result.difficulty}")
    print()


# ============================================================
# 演示 4：转 dict 和 JSON
# ============================================================

def demo_to_dict():
    """
    Pydantic 对象可以方便地转为 dict 或 JSON 字符串：
    - model_dump(): 转为 Python dict
    - model_dump_json(): 转为 JSON 字符串
    """
    class SimpleInfo(BaseModel):
        topic: str = Field(description="主题")
        summary: str = Field(description="一句话总结")

    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=SimpleInfo)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个总结助手。{format_instructions}"),
        ("human", "总结：LangChain 是一个 LLM 应用开发框架"),
    ])

    chain = prompt | llm | parser
    result = chain.invoke({
        "format_instructions": parser.get_format_instructions(),
    })

    print("=== Pydantic 对象转 dict / JSON ===")
    print(f"转 dict: {result.model_dump()}")
    print(f"转 JSON: {result.model_dump_json(indent=2)}")
    print()


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    demo_basic()
    demo_nested()
    demo_list_field()
    demo_to_dict()
