"""
04_few_shot_prompt.py
LangChain Prompts - Few-Shot Prompting (少样本提示)

Few-Shot Prompting 通过给模型提供"输入-输出"示例来引导模型按照特定格式或模式回答。
LangChain 提供了两个模板类来简化这个过程：

  - FewShotPromptTemplate              用于纯文本 prompt（旧版 LLM）
  - FewShotChatMessagePromptTemplate   用于 Chat Model 的消息列表

核心概念：
  - examples: 示例列表（输入-输出对）
  - example_prompt: 每个示例的模板
  - prefix: 示例之前的前缀文本
  - suffix: 示例之后的后缀文本

少样本学习的本质：让模型学会一种"格式"或"模式"，而不是学知识。
示例应该是模型不熟悉的任务格式（如自定义分类、特殊转换规则）。

参考文档：
  - FewShotPromptTemplate API: https://reference.langchain.com/python/langchain-core/prompts/few_shot/FewShotPromptTemplate
  - FewShotChatMessagePromptTemplate API: https://reference.langchain.com/python/langchain-core/prompts/few_shot/FewShotChatMessagePromptTemplate
  - 源码: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/prompts/few_shot.py

安装：
  pip install langchain-core zhipuai
"""

import os

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
    FewShotPromptTemplate,
)


# ============================================================
# 公共：模型实例
# ============================================================

def get_llm(temperature=0):
    """获取智谱 GLM 模型实例"""
    from langchain_community.chat_models import ChatTongyi
    llm = ChatTongyi(
        model="qwen-plus",
        dashscope_api_key=os.environ.get("ali_API_KEY"),
        temperature=temperature,
    )
    return llm


# ============================================================
# 演示 1：FewShotPromptTemplate（纯文本）
# ============================================================

def demo_few_shot_prompt():
    """
    FewShotPromptTemplate 用于旧版 LLM（纯文本补全）; 或者不需要区分消息类型的纯文本。
    它将 examples 按照格式插入到 prefix 和 suffix 之间。

    结构：prefix + 示例1 + 示例2 + ... + suffix   : 可参考演示 4：demo_structure

    这里演示一个"自定义格式转换"任务——给一个中文短语，输出拼音首字母缩写。
    这是模型不熟悉的具体格式规则，需要示例来教会它。
    """
    examples = [
        {"phrase": "人工智能", "abbreviation": "RGZN"},
        {"phrase": "机器学习", "abbreviation": "JQXX"},
    ]

    example_prompt = PromptTemplate.from_template("{phrase} -> {abbreviation}")

    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="将中文短语转换为拼音首字母缩写（大写）：",
        suffix="{input} ->",
        input_variables=["input"],
    )

    result = prompt.invoke({"input": "深度学习"})

    print("=== FewShotPromptTemplate（纯文本）===")
    print(result.to_string())




# ============================================================
# 演示 2：FewShotChatMessagePromptTemplate（Chat Model）
# ============================================================

def demo_few_shot_chat():
    """
    FewShotChatMessagePromptTemplate 用于 Chat Model。
    每个 example 是一组消息（human + ai），而不是单个字符串。

    演示任务：自定义评论分级（将用户评论分为 A/B/C 三级）。
    模型不知道我们的分级标准，需要示例来定义规则。
    """
    examples = [
        {
            "review": "产品很好用，物流也快，五星好评！",
            "grade": "A（正面满意）",
        },
        {
            "review": "东西还行，但包装有点破损。",
            "grade": "B（中性偏负面）",
        },
        {
            "review": "质量问题严重，完全不能用，申请退货。",
            "grade": "C（强烈负面）",
        },
    ]

    # 每个示例的消息模板
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "评论：{review}"),
        ("ai", "{grade}"),
    ])

    # 组装 few-shot 消息模板
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
    )


    # 将 few-shot 嵌入完整模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个评论分级助手。根据示例中的 A/B/C 分级标准，对新评论进行分级。"),
        few_shot_prompt,          # 这里会展开为多条消息
        ("human", "评论：{review}"),
    ])

    # 查看填充结果
    result = prompt.invoke({"review": "发货慢得要死，等了一周才到，但东西倒是没问题。"})

    print("=== FewShotChatMessagePromptTemplate ===")
    print(f"填充后的消息数: {len(result.to_messages())}")
    for msg in result.to_messages():
        print(f"  [{msg.type}] {msg.content}")
    print(result.to_string())


# ============================================================
# 演示 3：与 LLM 链式调用
# ============================================================

def demo_chain():
    """
    结合 LCEL 链式调用，few-shot 提示 → 模型生成。

    演示任务：将用户反馈转换为结构化的 JSON 格式。
    模型需要示例来学习具体的 JSON 结构和字段命名规则。
    """
    llm = get_llm()

    # 定义示例：反馈 → 结构化 JSON
    examples = [
        {
            "feedback": "登录页面加载太慢了，每次要等好几秒。",
            "output": '{"module": "登录", "type": "性能问题", "severity": "中", "summary": "登录页面加载缓慢"}',
        },
        {
            "feedback": "搜索结果经常不准，搜A出来B。",
            "output": '{"module": "搜索", "type": "功能缺陷", "severity": "高", "summary": "搜索结果不准确"}',
        },
    ]

    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "用户反馈：{feedback}"),
        ("ai", "{output}"),
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是反馈分析助手。将用户反馈转换为 JSON，包含 module/type/severity/summary 四个字段。只输出 JSON，不要解释。"),
        few_shot_prompt,
        ("human", "用户反馈：{feedback}"),
    ])

    chain = prompt | llm

    result = chain.invoke({"feedback": "购物车页面点结算按钮没反应，试了好几次都不行。"})
    print(f"=== 链式调用：反馈结构化 ===")
    print(f"反馈: 购物车页面点结算按钮没反应，试了好几次都不行。")
    print(f"结果: {result.content}")



# ============================================================
# 演示 4：FewShotPromptTemplate 的 prefix/suffix 结构
# ============================================================

def demo_structure():
    """
    FewShotPromptTemplate 的内部结构：

    prefix（前缀）
      ↓
    example 1（示例 1）
    example 2（示例 2）
    ...
      ↓
    suffix（后缀 + 用户输入）

    最终输出 = prefix + formatted_examples + suffix
    """
    examples = [
        {"text": "今天天气真好", "sentiment": "正面"},
        {"text": "服务态度太差了", "sentiment": "负面"},
    ]

    example_prompt = PromptTemplate.from_template("文本: {text}\n情感: {sentiment}")

    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="请根据示例分析文本的情感倾向（正面/负面/中性）：\n",
        suffix="\n文本: {input}\n情感:",
        input_variables=["input"],
    )

    result = prompt.invoke({"input": "这款产品一般般吧"})

    print("=== FewShotPromptTemplate 内部结构 ===")
    print("prefix: 请根据示例分析文本的情感倾向（正面/负面/中性）：")
    print("examples:")
    print("  文本: 今天天气真好  → 情感: 正面")
    print("  文本: 服务态度太差了  → 情感: 负面")
    print("suffix: 文本: {input} → 情感:")
    print()
    print("--- 实际输出 ---")
    print(result.to_string())



# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    if not os.environ.get("ali_API_KEY"):
        print("请设置 ali_API_KEY 环境变量")
        print("  export ali_API_KEY='...'")
        exit(1)

    demo_few_shot_prompt()
    demo_few_shot_chat()
    demo_chain()
    demo_structure()