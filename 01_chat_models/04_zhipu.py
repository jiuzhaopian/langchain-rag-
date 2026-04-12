"""
04_zhipu.py
智谱 GLM 官方集成 - ChatZhipuAI

使用 langchain-community 官方包的 ChatZhipuAI 类。
这是 LangChain 官方推荐的智谱集成方式。

参考文档：
  - 官方集成文档: https://docs.langchain.com/oss/python/integrations/chat/zhipuai
  - API Key: https://open.bigmodel.cn/usercenter/apikeys

安装：
  pip install langchain-community
  pip install zhipuai
  pip install pyjwt
"""

import os


def demo_zhipu():
    """
    使用 ChatZhipuAI（官方集成）
    参考: https://docs.langchain.com/oss/python/integrations/chat/zhipuai
    """
    from langchain_community.chat_models import ChatZhipuAI

    # 初始化
    chat = ChatZhipuAI(
        model="glm-4.7",
        #model="glm-4-flash",       # 免费模型
        # model="glm-4",           # 旗舰模型
        # model="glm-4-plus",      # 增强模型
        temperature=0.5,
    )

    # invoke
    response = chat.invoke("用一句话介绍 LangChain")

    # 输出完整响应 JSON
    print(response.model_dump_json(indent=2))
    print(f"[ChatZhipuAI invoke] {response.content}")
    print(f"类型: {type(response).__name__}")
    print(f"模型: {response.response_metadata.get('model_name', 'N/A')}")
    print(f"Token: {response.usage_metadata}")


if __name__ == "__main__":
    if not os.environ.get("ZHIPUAI_API_KEY"):
        print("请设置 ZHIPUAI_API_KEY 环境变量")
        print("  export ZHIPUAI_API_KEY='...'")
        print("\nAPI Key 获取: https://open.bigmodel.cn/usercenter/apikeys")
        exit(1)

    print("=== 智谱 GLM 官方集成 (ChatZhipuAI) ===\n")
    demo_zhipu()
