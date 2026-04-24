"""
openai_compatible.py
    使用openai sdk 完成 ai对话

    依赖：
        pip install -U openai
"""
from openai import OpenAI
from dotenv import load_dotenv # 需要先安装：pip install -U dotenv

load_dotenv(dotenv_path="../.env")

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    #api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

if __name__ == '__main__':

    # 对话
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "请模拟韩立的身份回答用户的任何问题"},
            {"role": "user", "content": "你是谁？"},
        ]
    )
    print(completion.model_dump_json())