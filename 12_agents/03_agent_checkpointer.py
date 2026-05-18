"""
03_agent_checkpointer.py
LangChain Agents - Checkpointer（会话记忆）

Checkpointer 在每步执行后保存状态快照，让 Agent 具备会话记忆。

对比多轮对话的两种方式：
  1. 手动传消息历史（01 演示 4 的做法）—— 需要自己管理消息列表
  2. Checkpointer —— 通过 thread_id 自动保存和恢复状态

核心概念：
  - Thread：通过 thread_id 标识的会话，同一 thread 内的 invoke 共享状态
  - Checkpoint：每步执行后的状态快照，支持时间旅行和故障恢复
  - thread_id 是必须的，没有它 checkpointer 无法保存或恢复状态

参考文档：
  - Persistence 概述: https://docs.langchain.com/oss/python/langgraph/persistence
  - Add Memory: https://docs.langchain.com/oss/python/langgraph/add-memory
  - Time Travel: https://docs.langchain.com/oss/python/langgraph/use-time-travel

安装：
  pip install langchain langgraph langgraph-checkpoint-sqlite langchain-openai python-dotenv

注意：本文件使用 ChatOpenAI 连接 DashScope（阿里云百炼）兼容接口，
不需要 OpenAI API Key，只需要 DASHSCOPE_API_KEY。
"""

import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


# ========================= 模型配置 =========================

model = ChatOpenAI(
    model="qwen3.6-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
)


# ========================= 工具定义 =========================

@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积"""
    return a * b


# ============================================================
# 演示 1：MemorySaver —— 基础会话记忆
# ============================================================

def demo_memory_saver():
    """
    MemorySaver 是内存版本的 checkpointer，适合开发和测试。
    进程重启后数据丢失。

    关键：每次 invoke 都需要传 config（包含 thread_id）。
    同一 thread_id 的调用共享记忆，不同 thread_id 互不影响。
    """
    print(">>> 演示 1：MemorySaver —— 基础会话记忆")
    print("-" * 50)

    from langgraph.checkpoint.memory import MemorySaver

    agent = create_agent(model, tools=[add, multiply], checkpointer=MemorySaver())

    config = {
        "configurable": {
            "thread_id": "demo-session-1"
        }
    }

    # 第一轮：告诉 Agent 一个事实
    result1 = agent.invoke(
        {"messages": [HumanMessage(content="我刚才说的数字是 42，帮我记住它")]},
        config=config,
    )
    print(f"第一轮: {result1['messages'][-1].content}")

    # 第二轮：新 invoke，不带历史消息，但同一个 thread_id → 有记忆
    result2 = agent.invoke(
        {"messages": [HumanMessage(content="我刚才说的数字是什么？帮我乘以 2")]},
        config=config,
    )
    print(f"第二轮（同一 thread）: {result2['messages'][-1].content}")

    # 第三轮：不同 thread_id → 没有记忆
    config_new = {
        "configurable": {
            "thread_id": "demo-session-2"
        }
    }
    result3 = agent.invoke(
        {"messages": [HumanMessage(content="我刚才说的数字是什么？")]},
        config=config_new,
    )
    print(f"新会话（不同 thread）: {result3['messages'][-1].content}")
    print()


# ============================================================
# 演示 2：SqliteSaver —— 持久化会话记忆
# ============================================================

def demo_sqlite_saver():
    """
    SqliteSaver 将状态保存到 SQLite 文件，进程重启后数据不丢失。
    适合单机部署、中小规模应用。

    用法：SqliteSaver.from_conn_string("path/to/db.sqlite")
    文件不存在时自动创建。
    """
    print(">>> 演示 2：SqliteSaver —— 持久化会话记忆")
    print("-" * 50)

    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path = os.path.join(os.path.dirname(__file__), "data", "checkpoint.sqlite")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    with SqliteSaver.from_conn_string(db_path) as checkpointer:

        agent = create_agent(model, tools=[add, multiply], checkpointer=checkpointer)

        config = {
            "configurable": {
                "thread_id": "sqlite-demo"
            }
        }

        result1 = agent.invoke(
            {"messages": [HumanMessage(content="记住：我最喜欢的数字是 88")]},
            config=config,
        )
        print(f"第一轮: {result1['messages'][-1].content}")

        # 模拟"重启"：重新创建 checkpointer，数据从文件恢复
        with SqliteSaver.from_conn_string(db_path) as cp2:
            agent2 = create_agent(model, tools=[add, multiply], checkpointer=cp2)
            result2 = agent2.invoke(
                {"messages": [HumanMessage(content="我最喜欢的数字是什么？")]},
                config=config,
            )
            print(f"重新连接后（从 SQLite 恢复）: {result2['messages'][-1].content}")

    print()


# ============================================================
# Checkpointer 持久化方案速查
# ============================================================

CHECKPOINTER_REFERENCE = """
Checkpointer 持久化方案速查：

┌──────────────────────┬─────────────────┬──────────────────────────────────────────┐
│ 方案                 │ 适用场景         │ 安装 & 用法                                │
├──────────────────────┼─────────────────┼──────────────────────────────────────────┤
│ MemorySaver          │ 开发/测试        │ 内置，无需安装                              │
│                      │                 │ from langgraph.checkpoint.memory import   │
│                      │                 │ MemorySaver                              │
├──────────────────────┼─────────────────┼──────────────────────────────────────────┤
│ SqliteSaver          │ 单机/中小项目    │ pip install langgraph-checkpoint-sqlite   │
│                      │                 │ from langgraph.checkpoint.sqlite import   │
│                      │                 │ SqliteSaver                              │
├──────────────────────┼─────────────────┼──────────────────────────────────────────┤
│ PostgresSaver        │ 生产/多实例部署   │ pip install langgraph-checkpoint-postgres│
│                      │                 │ from langgraph.checkpoint.postgres import│
│                      │                 │ PostgresSaver                            │
├──────────────────────┼─────────────────┼──────────────────────────────────────────┤
│ AsyncSqliteSaver     │ 异步场景         │ 同 sqlite 包，aio 子模块                   │
│ AsyncPostgresSaver   │ 异步+生产        │ 同 postgres 包，aio 子模块                │
└──────────────────────┴─────────────────┴──────────────────────────────────────────┘

社区维护的 Checkpointer：
  - Redis:   pip install langgraph-checkpoint-redis
  - MongoDB: pip install langgraph-checkpoint-mongodb

如何发现更多 Checkpointer 方案：
  1. PyPI 搜索: https://pypi.org/search/?q=langgraph-checkpoint
  2. GitHub: https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint
  3. 官方文档: https://docs.langchain.com/oss/python/langgraph/persistence#checkpointer-options

参考文档：
  - Persistence 概述:  https://docs.langchain.com/oss/python/langgraph/persistence
  - Add Memory:         https://docs.langchain.com/oss/python/langgraph/add-memory
  - Time Travel:        https://docs.langchain.com/oss/python/langgraph/use-time-travel
"""


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("错误：请设置 DASHSCOPE_API_KEY 环境变量")
        exit(1)

    demo_memory_saver()
    demo_sqlite_saver()

    print(CHECKPOINTER_REFERENCE)