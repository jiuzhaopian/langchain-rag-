import os

# 测试环境变量是否读取成功
print(os.getenv("deepseek_API_KEY"))  # 应该打印出你的密钥，而不是 None
print(os.getenv("ali_API_KEY"))       # 应该打印出你的阿里密钥

# 如果打印出 None，说明环境变量没有生效