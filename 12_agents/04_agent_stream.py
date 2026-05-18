"""
04_agent_stream.py
LangChain Agents - 流式输出（Streaming）

Agent 基于 LangGraph 构建，天然支持流式输出。
stream() 方法支持 7 种 stream_mode，本文件聚焦最常用的 "messages" 模式。

企业实践中，"messages" 是唯一需要掌握的流式模式——
它实现打字机效果，是所有 chat 类产品的标配交互方式。
其他 6 种模式是调试/监控工具，放在速查表中供参考。

参考文档：
  - Streaming 概述: https://docs.langchain.com/oss/python/langgraph/streaming
  - Persistence: https://docs.langchain.com/oss/python/langgraph/persistence

安装：
  pip install langchain langgraph langchain-openai python-dotenv

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
    streaming=True,  # 启用流式 token 输出（stream_mode="messages" 需要）
)


# ========================= 工具定义 =========================

@tool
def search_product(name: str) -> str:
    """在商品库中搜索商品，返回商品名称和价格"""
    products = {
        "iPhone": {"name": "iPhone 16 Pro", "price": 8999},
        "MacBook": {"name": "MacBook Pro 14", "price": 14999},
        "AirPods": {"name": "AirPods Pro 3", "price": 1899},
    }
    for key, val in products.items():
        if key.lower() in name.lower():
            return f"{val['name']}，价格: {val['price']} 元"
    return f"未找到与 '{name}' 相关的商品"

@tool
def check_inventory(product_name: str) -> str:
    """查询商品的库存数量"""
    inventory = {"iPhone 16 Pro": 50, "MacBook Pro 14": 20, "AirPods Pro 3": 200}
    stock = inventory.get(product_name, 0)
    return f"{product_name} 库存: {stock} 台"

tools = [search_product, check_inventory]


# ============================================================
# 核心演示：stream_mode="messages" —— 打字机效果
# ============================================================

def demo_messages():
    """
    stream_mode="messages"：LLM 消息逐 chunk 产出。
    每个元素是 (message_chunk, metadata) 元组。

    这是企业实践中唯一需要掌握的流式模式：
    - 前端展示打字机效果
    - 实时显示 AI 的回复过程
    - 所有 chat 类产品的标配交互方式

    注意：
    - AIMessage 的 content 会被分成多个 chunk，每个 chunk 包含几个字符
      （具体粒度取决于模型的 streaming 实现，不一定是逐 token）
    - qwen3.6-plus 在思考阶段会产生大量 content=None 的 chunk，需要过滤
    - 模型配置需要 streaming=True 才能产出 chunk 级输出
    """
    print(">>> stream_mode='messages' —— 打字机效果")
    print("-" * 50)

    agent = create_agent(
        model,
        tools=tools,
        system_prompt="你是一个电商助手，用五句话回答。不要调用工具，直接回答。"
    )

    # 用问题让模型直接回答（不调工具），演示 chunk 级流式
    print("  AI 回复: ", end="", flush=True)
    chunk_count = [0]
    for msg, metadata in agent.stream(
        {"messages": [HumanMessage(content="介绍一下你们的店铺特色")]},
        stream_mode="messages",
    ):
        msg_type = type(msg).__name__
        if msg_type in ("AIMessage", "AIMessageChunk") and msg.content:
            # content 是文本片段，逐个输出形成打字机效果
            # qwen3.6-plus 的思考阶段 content 为 None，已自动过滤
            print(msg.content, end="", flush=True)
            chunk_count[0] += 1
    print()
    print(f"  （共收到 {chunk_count[0]} 个有效 chunk）")
    print()

    # --- metadata 结构说明 ---
    # 每个 (message_chunk, metadata) 元组的 metadata 包含：
    #   langgraph_step: 1              -- 当前执行到第几步
    #   langgraph_node: "model"        -- 当前节点名（model/tools）
    #   langgraph_triggers: (...)      -- 触发当前节点的原因
    #   langgraph_path: (...)          -- 执行路径
    #   checkpoint_ns: "model:xxx"     -- checkpoint 命名空间
    #   ls_provider: "openai"          -- 模型提供者
    #   ls_model_name: "qwen3.6-plus"  -- 模型名
    #   ls_model_type: "chat"          -- 模型类型


# ============================================================
# Stream Mode 速查表
# ============================================================

STREAM_MODE_REFERENCE = """
Stream Mode 速查（7 种模式）

来源：https://docs.langchain.com/oss/python/langgraph/streaming

agent.stream() 的 stream_mode 参数，决定流式输出的粒度。

┌──────────────┬────────────────────────────┬──────────────────────────┐
│ stream_mode  │ data 类型                 │ 说明                     │
├──────────────┼────────────────────────────┼──────────────────────────┤
│ "values"     │ 完整 state (dict)          │ 每步后的完整状态快照       │
│ "updates"    │ {node: update} (dict)      │ 每步后的状态变化（增量）   │
│ "messages"   │ (MessageChunk, metadata)   │ ⭐ LLM token + metadata │
├──────────────┼────────────────────────────┼──────────────────────────┤
│ "custom"     │ 任意数据                   │ 节点内 get_stream_writer  │
│ "checkpoints"│ checkpoint 事件            │ 同 get_state() 格式       │
│ "tasks"      │ 任务开始/结束事件           │ 含 results 和 errors     │
│ "debug"      │ 全量信息                   │ checkpoints + tasks + 额外元数据 │
└──────────────┴────────────────────────────┴──────────────────────────┘

要点：
  - stream_mode 不传时默认为 "values"
  - 可以组合：stream_mode=["updates", "messages"]
  - 组合时建议加 version="v2"，返回统一的 {"type": ..., "data": ...} 格式
  - "messages" 只产出 LLM 调用的 token，不含 ToolMessage

--- values 示例（每步输出完整 state）---

# values: 每步返回完整的 state（含所有 key 的当前值）
# 例：3 个节点执行后，你会收到 3 个 state 快照
for chunk in agent.stream(input, stream_mode="values", version="v2"):
    if chunk["type"] == "values":
        print(chunk["data"])  # 完整 state dict

--- updates 示例（每步只输出变化部分）---

# updates: 每步返回 {node_name: 该节点的更新}（只有变化的 key）
for chunk in agent.stream(input, stream_mode="updates", version="v2"):
    if chunk["type"] == "updates":
        for node_name, state in chunk["data"].items():
            print(f"Node `{node_name}` updated: {state}")

选择建议：
  - 上产品 / 做前端打字机 → "messages"
  - 调试执行流程 → "values" 或 "updates"
  - 其他 → 按需查上面链接
"""


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("错误：请设置 DASHSCOPE_API_KEY 环境变量")
        exit(1)

    demo_messages()

    print(STREAM_MODE_REFERENCE)
