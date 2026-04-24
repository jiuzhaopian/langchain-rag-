"""
01_prompt_template.py
LangChain Prompts - PromptTemplate (纯文本模板)

PromptTemplate 用于生成纯文本 prompt，主要用于旧版 LLM（纯文本补全）。
Chat Model 场景下应使用 ChatPromptTemplate（见 02_chat_prompt_template.py）。

核心概念：
  - 模板字符串中用 {variable} 作为占位符
  - invoke({"variable": "值"}) 填充变量，生成最终字符串
  - 输入类型是 dict，输出类型是 StringPromptValue

参考文档：
  - PromptTemplate API: https://reference.langchain.com/python/langchain-core/prompts/prompt/PromptTemplate
  - 源码: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/prompts/prompt.py

安装：
  pip install langchain-core
"""

from datetime import datetime

from langchain_core.prompts import PromptTemplate


# ============================================================
# 演示 1：基本用法
# ============================================================

def demo_basic():
    """
    最简单的 PromptTemplate：一个模板字符串 + 变量列表
    """
    pass


# ============================================================
# 演示 2：多个变量
# ============================================================

def demo_multiple_variables():
    """
    模板中可以有多个变量，invoke 时必须全部提供
    """
    template = PromptTemplate(
        template="你是一个{role}，请用{style}的风格回答问题。\n\n问题：{question}",
        input_variables=["role", "style", "question"],
    )

    result = template.invoke({
        "role": "Python 专家",
        "style": "通俗易懂",
        "question": "什么是装饰器？",
    })

    print("=== 多个变量 ===")
    print(result.to_string())
    print()


# ============================================================
# 演示 3：from_template 简写
# ============================================================

def demo_from_template():
    """
    from_template() 自动从模板字符串中提取变量名，无需手写 input_variables
    """
    pass


# ============================================================
# 演示 4：partial 预填充变量
# ============================================================

def demo_partial():
    """
    partial() 可以预先填充部分变量，返回一个只需要剩余变量的新模板。
    适用于：某些变量是固定的（如角色、风格），只有部分需要动态传入。
    """
    template = PromptTemplate.from_template(
        "你是{name}，请用{tone}的语气回答：{question}"
    )

    # 预填充 name 和 tone
    partial_template = template.partial(name="小助手", tone="友好")

    # 现在只需要传 question
    result = partial_template.invoke({"question": "今天天气怎么样？"})

    print("=== partial 预填充 ===")
    print(f"原始变量: {template.input_variables}")
    print(f"预填充 name='小助手', tone='友好'")
    print(f"填充结果: {result.to_string()}")
    print()


# ============================================================
# 演示 5：带默认值的变量（partial_variables）
# ============================================================

def demo_partial_variables():
    """
    partial_variables 在创建模板时就绑定默认值，
    invoke 时可以覆盖，也可以不传。
    """
    template = PromptTemplate(
        template="当前时间：{current_time}\n用户问题：{question}",
        input_variables=["question"],
        partial_variables={"current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
    )

    result = template.invoke({"question": "你好"})

    print("=== partial_variables 默认值 ===")
    print(result.to_string())
    print()


# ============================================================
# 演示 6：输出类型 StringPromptValue
# ============================================================

def demo_output_types():
    """
    PromptTemplate.invoke() 返回 StringPromptValue，
    它有两种转换方式：
      - to_string() → 纯文本字符串
      - to_messages() → [HumanMessage]（包装成消息列表）
    """
    template = PromptTemplate.from_template("解释一下{concept}")
    result = template.invoke({"concept": "量子计算"})

    print("=== 输出类型 ===")
    print(f"类型: {type(result).__name__}")
    print(f"to_string() 类型: {type(result.to_string()).__name__}")
    print(f"to_string(): {result.to_string()}")
    messages = result.to_messages()
    print(f"to_messages() 类型: {[type(m).__name__ for m in messages]}")
    print(f"to_messages()[0].content: {messages[0].content}")
    print(f"to_messages()[0].type: {messages[0].type}")
    print()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    demo_basic()
    demo_multiple_variables()
    demo_from_template()
    demo_partial()
    demo_partial_variables()
    demo_output_types()
