"""04_custom_parser.py - Custom Parser (自定义解析器)

当内置 Parser 不满足需求时，继承 BaseOutputParser 自定义。

核心步骤：
  1. 继承 BaseOutputParser[T]，T 是输出类型（如 dict, list[str]）
  2. 实现 parse() 方法（必须）—— 定义如何将文本转为目标类型
  3. 可选：实现 get_format_instructions() —— 注入格式指令让 LLM 按指定格式输出
  4. 可选：实现 _type 属性 —— 返回 parser 的类型标识字符串，用于日志和序列化

适合场景：
  - 自定义格式（如 "key: value" 每行一条）
  - 后处理（如提取特定模式、转换数据）
  - 与遗留系统集成（如特定分隔符格式）

参考文档：
  - BaseOutputParser API: https://reference.langchain.com/python/langchain-core/output_parsers/base/BaseOutputParser
  - 源码: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/output_parsers/base.py

安装：
  pip install langchain-core langchain-community zhipuai
"""

import re
from typing import List
from langchain_core.output_parsers import BaseOutputParser
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
# 自定义 Parser 1：键值对解析器
# ============================================================

class KeyValueParser(BaseOutputParser[dict]):
    """
    将 "key: value" 每行一条的格式解析为 dict。
    示例输入：
      name: 张三
      age: 25
      city: 北京
    """

    def parse(self, text: str) -> dict:
        """按行读取，用冒号分隔每行的 key 和 value"""
        result = {}
        for line in text.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                result[key.strip()] = value.strip()
        return result

    def get_format_instructions(self) -> str:
        return (
            "请按以下格式输出，每行一个键值对，用冒号分隔：\n"
            "key1: value1\n"
            "key2: value2\n"
            "key3: value3"
        )

    @property
    def _type(self) -> str:
        # parser 的类型标识，用于日志和序列化，返回自定义字符串即可
        return "key_value_parser"


def demo_key_value():
    """演示自定义键值对解析器"""
    llm = get_llm()
    parser = KeyValueParser()

    print("=== KeyValueParser 格式指令 ===")
    print(parser.get_format_instructions())
    print()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个信息提取助手。{format_instructions}"),
        ("human", "提取：姓名是张三，年龄25岁，职业是程序员，城市北京"),
    ])

    chain = prompt | llm | parser
    result = chain.invoke({
        "format_instructions": parser.get_format_instructions(),
    })

    print("=== KeyValueParser 结果 ===")
    print(f"类型: {type(result).__name__}")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print()


# ============================================================
# 自定义 Parser 2：评分提取器
# ============================================================

class ScoreExtractor(BaseOutputParser[dict]):
    """
    从 LLM 输出中提取评分和理由。
    期望格式：
      评分: X
      理由: xxxxx

    parse 原理：用正则表达式从文本中匹配 "评分:" 和 "理由:" 后面的内容。
    - r"评分[：:]\\s*(\\d+(?:\\.\\d+)?)" 匹配 "评分" 后跟中英文冒号、空格、然后是数字（支持小数）
    - r"理由[：:]\\s*(.+)" 匹配 "理由" 后跟中英文冒号、空格、然后取剩余全部内容
    同时兼容中文冒号和英文冒号，因为 LLM 输出可能混用。
    """

    def parse(self, text: str) -> dict:
        # 匹配评分：中英文冒号后跟数字（支持小数如 8.5）
        score_match = re.search(r"评分[：:]\s*(\d+(?:\.\d+)?)", text)
        # 匹配理由：中英文冒号后取剩余全部内容
        reason_match = re.search(r"理由[：:]\s*(.+)", text)

        score = float(score_match.group(1)) if score_match else 0.0
        reason = reason_match.group(1).strip() if reason_match else ""

        return {"score": score, "reason": reason}

    def get_format_instructions(self) -> str:
        return (
            "请按以下格式输出你的评分：\n"
            "评分: 1-10 的数字\n"
            "理由: 一句话说明理由"
        )

    @property
    def _type(self) -> str:
        return "score_extractor"


def demo_score_extractor():
    """演示自定义评分提取器"""
    llm = get_llm()
    parser = ScoreExtractor()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个产品评分助手。{format_instructions}"),
        ("human", "评价 iPhone 16"),
    ])

    chain = prompt | llm | parser
    result = chain.invoke({
        "format_instructions": parser.get_format_instructions(),
    })

    print("=== ScoreExtractor 结果 ===")
    print(f"评分: {result['score']}")
    print(f"理由: {result['reason']}")
    print()


# ============================================================
# 自定义 Parser 3：带容错的 JSON 解析器
# ============================================================

class RobustJsonParser(BaseOutputParser[dict]):
    """
    带容错的 JSON 解析器。
    如果标准 JSON 解析失败，尝试去除 markdown 代码块标记后再解析。
    """

    def parse(self, text: str) -> dict:
        import json

        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试去除 markdown 代码块标记（LLM 经常用 ```json ... ``` 包裹输出）
        cleaned = re.sub(r"```json?\s*", "", text)
        cleaned = re.sub(r"```", "", cleaned)
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # 最后尝试提取花括号内容
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"无法解析为 JSON: {text[:100]}...")

    @property
    def _type(self) -> str:
        return "robust_json_parser"


def demo_robust_json():
    """演示带容错的 JSON 解析器（不调用 LLM，直接测试 parse）"""
    parser = RobustJsonParser()

    # 测试 1：正常 JSON
    r1 = parser.parse('{"name": "test", "value": 42}')
    print("=== 测试 1：正常 JSON ===")
    print(f"结果: {r1}")
    print()

    # 测试 2：带 markdown 代码块的 JSON
    r2 = parser.parse('```json\n{"name": "test", "value": 42}\n```')
    print("=== 测试 2：带 markdown 代码块 ===")
    print(f"结果: {r2}")
    print()

    # 测试 3：JSON 前后有额外文字
    r3 = parser.parse('这是结果：\n{"name": "test", "value": 42}\n以上是结果。')
    print("=== 测试 3：JSON 前后有额外文字 ===")
    print(f"结果: {r3}")
    print()


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    demo_key_value()
    demo_score_extractor()
    demo_robust_json()
