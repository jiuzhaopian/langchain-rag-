"""
04_chain_principle.py - | 运算符的底层原理

LCEL 的 | 管道符为什么能把 prompt | llm | parser 串联起来？
核心在于 Python 的运算符重载和链式设计模式。

本文件从零实现一个"迷你版 LCEL"，帮助理解原理：
  - MyRunnable: 顶层基类，定义 invoke 方法（类似 LCEL 的 Runnable）
  - MySequence: 串联多个 MyRunnable（类似 LCEL 的 RunnableSequence）
  - 通过子类重写 invoke，直观看到数据如何流经每个节点

不需要安装任何 LangChain 依赖，本文件是纯 Python 代码，直接运行即可。
"""


# ============================================================
# 核心机制
# ============================================================
#
# Python 运算符重载规则：
#   a | b  ->  先调用 a.__or__(b)，如果返回 NotImplemented 再调用 b.__ror__(a)
#
# 所以 a | b 的关键是让 __or__ 返回一个"链"对象，而不是单个节点。

# TODO
class MyRunnable:
    """顶层基类，类似 LCEL 的 Runnable。
    所有节点和链都继承它，提供统一的 invoke 方法。
    """



class MySequence(MyRunnable):
    """处理链，类似 LCEL 的 RunnableSequence。
    每收到一个 | 就追加一个节点，执行时按顺序依次调用。
    """


# ============================================================
# 演示：用具体节点看 | 串联的执行过程
# ============================================================

def demo():
    # 定义三个处理节点，每个都继承 MyRunnable

    class UpperNode(MyRunnable):
        """把输入转为大写"""
        def invoke(self, input_data):
            result = input_data.upper()
            print(f"  [大写] {input_data} -> {result}")
            return result

    class AddBangNode(MyRunnable):
        """在末尾加感叹号"""
        def invoke(self, input_data):
            result = input_data + "!"
            print(f"  [加感叹号] {input_data} -> {result}")
            return result

    class AddPrefixNode(MyRunnable):
        """加前缀"""
        def invoke(self, input_data):
            result = f"结果: {input_data}"
            print(f"  [加前缀] {input_data} -> {result}")
            return result

    # | 串联
    chain = UpperNode() | AddBangNode() | AddPrefixNode()
    print(f"chain = {chain}")
    print(f"type:  {type(chain).__name__}")
    print()

    # 执行
    print('chain.invoke("hello")')
    print("--- 执行过程 ---")
    result = chain.invoke("hello")
    print("--- 最终结果 ---")
    print(f"  {result}")
    print()

    # 对应关系
    print("--- 与 LCEL 的对应关系 ---")
    print("MyRunnable    -> Runnable          (顶层基类，定义 invoke)")
    print("MySequence    -> RunnableSequence   (串联多个 Runnable)")
    print("__or__        -> | 运算符重载       (把节点串成链)")
    print(".invoke()     -> .invoke()          (执行处理)")
    print()


if __name__ == "__main__":
    demo()
