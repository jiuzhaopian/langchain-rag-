"""
02_parallel_branch.py - RunnableParallel 并行 + RunnableBranch 条件分支

Runnable 组合模式：
  - RunnableSequence（| 管道）：串行，上一个输出是下一个输入
  - RunnableParallel（| 管道中的 dict）：并行，同一输入分发给多个 Runnable
  - RunnableBranch：条件分支，根据输入选择不同的 Runnable 执行

参考文档：
  - RunnableParallel: https://reference.langchain.com/python/langchain-core/runnables/base/RunnableParallel
  - RunnableBranch: https://reference.langchain.com/python/langchain-core/runnables/branch/RunnableBranch
  - RunnablePassthrough: https://reference.langchain.com/python/langchain-core/runnables/passthrough/RunnablePassthrough
  - RunnableLambda: https://reference.langchain.com/python/langchain-core/runnables/base/RunnableLambda
  - 源码: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/runnables/

安装：
  pip install langchain-core langchain-community zhipuai
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough, RunnableParallel


# ========================= 模型配置 =========================

def get_llm():
    from langchain_community.chat_models import ChatZhipuAI
    import os
    return ChatZhipuAI(
        model="glm-4.7",
        api_key=os.environ.get("ZHIPUAI_API_KEY"),
    )


# ============================================================
# 演示 1：RunnableParallel 并行执行
# ============================================================

def demo_parallel():
    """
    RunnableParallel 接收一个输入，分发给多个 Runnable 并行执行。
    结果是 dict，key 是你起的名字，value 是对应 Runnable 的输出。

    构造方式：
      1. 显式构造：RunnableParallel(cn=..., en=...)
      2. | 管道中的 dict：自动转为 RunnableParallel
         Runnable 同时实现了 __or__ 和 __ror__，所以 dict 在 | 的左侧或右侧都能自动转换：
           prompt | {"k": v}  -> prompt.__or__(dict)  -> coerce 右侧
           {"k": v} | prompt  -> prompt.__ror__(dict)  -> coerce 左侧
         注意：直接赋值给变量（chain = {"k": r1}）仍然是 dict，不能调用 .invoke()。
    """
    llm = get_llm()
    
    # ---- 方式 1：显式构造 ----
    # TODO
    pass

    # ---- 方式 2a：dict 在 | 右侧 ----
    # StrOutputParser 的输出（str）分发给两个分支
    print("=== 方式 2a：dict 在 | 右侧 ===")
    # TODO
    pass

    # ---- 方式 2b：dict 在 | 左侧 ----
    # 同一个输入分发给两个不同的 prompt+llm 链
    print("=== 方式 2b：dict 在 | 左侧 ===")
    # TODO
    pass

    # ---- 常见错误示范 ----
    # wrong = {"key": llm}
    # wrong.invoke(...)  # AttributeError: 'dict' object has no attribute 'invoke'
    # 原因：直接赋值给变量的 dict 不会被自动转换，必须是 | 运算符触发的


# ============================================================
# 演示 2：RunnableBranch 条件分支
# ============================================================

def demo_branch():
    """
    RunnableBranch 根据条件选择不同的 Runnable 执行，类似 if-elif-else。

    注意事项：
      - 条件函数接收的是整个输入（通常是一个 dict）
      - 必须返回 True/False（不是 truthy/falsy，是布尔值）
      - 最后一个参数是默认分支，不需要条件
    """
    llm = get_llm()

    math_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "你是一个数学老师，请解答数学题"),
            ("human", "{input}"),
        ]) | llm | StrOutputParser()
    )

    code_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "你是一个编程助手，请回答编程问题"),
            ("human", "{input}"),
        ]) | llm | StrOutputParser()
    )

    general_chain = (
        ChatPromptTemplate.from_messages([
            ("human", "{input}"),
        ]) | llm | StrOutputParser()
    )

    # 条件函数接收 dict，从中取 "input" 字段判断
    def is_math(x: dict) -> bool:
        text = x.get("input", "")
        return "计算" in text or "多少" in text

    def is_code(x: dict) -> bool:
        text = x.get("input", "")
        return "代码" in text or "编程" in text or "bug" in text.lower()

    branch = RunnableBranch(
        (is_math, math_chain),
        (is_code, code_chain),
        general_chain,  # 默认分支
    )

    print("=== RunnableBranch ===")
    r1 = branch.invoke({"input": "计算 3 + 5 * 2 等于多少"})
    print(f"数学题 -> {r1}")

    r2 = branch.invoke({"input": "Python 怎么写一个 for 循环"})
    print(f"编程题 -> {r2}")

    r3 = branch.invoke({"input": "你好，今天天气怎么样"})
    print(f"通用 -> {r3}")
    print()


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    demo_parallel()
    demo_branch()
