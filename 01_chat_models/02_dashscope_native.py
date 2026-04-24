"""
02_dashscope_native.py
通义千问 DashScope 原生接口 - ChatTongyi

与 OpenAI 兼容模式的区别：
  - OpenAI 兼容模式（01_openai_compatible.py）：用 ChatOpenAI + base_url，走 OpenAI 协议
  - 原生模式（本文件）：用 ChatTongyi，走 DashScope 自己的协议

ChatTongyi 优势：
  - 支持通义千问独有功能（如通义万相图片生成）
  - 原生支持 tool calling
  - 默认模型为 qwen-turbo（速度快、成本低）

参考文档：
  - ChatTongyi 集成: https://python.langchain.com/docs/integrations/chat/tongyi/

安装：
  pip install langchain_community
  pip install dashscope
"""

import os

def demo_chat_tongyi():
    """
    使用 ChatTongyi（通义千问原生接口）
    参考: https://python.langchain.com/docs/integrations/chat/tongyi/
    """
    from langchain_community.chat_models import ChatTongyi
    #TODO
    pass


if __name__ == "__main__":
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("请设置 DASHSCOPE_API_KEY 环境变量")
        print("  export DASHSCOPE_API_KEY='sk-...'")
        exit(1)

    print("=== ChatTongyi（通义千问原生接口） ===\n")
    demo_chat_tongyi()
