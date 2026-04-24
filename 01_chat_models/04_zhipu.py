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
    #TODO
    pass


if __name__ == "__main__":
    if not os.environ.get("ZHIPUAI_API_KEY"):
        print("请设置 ZHIPUAI_API_KEY 环境变量")
        print("  export ZHIPUAI_API_KEY='...'")
        print("\nAPI Key 获取: https://open.bigmodel.cn/usercenter/apikeys")
        exit(1)

    print("=== 智谱 GLM 官方集成 (ChatZhipuAI) ===\n")
    demo_zhipu()
