"""
05_ollama.py
本地模型 Ollama - 无需 API Key，完全本地运行

前提条件：
  1. 安装 Ollama: https://ollama.com/download
     macOS: brew install ollama && brew services start ollama
  2. 拉取模型: ollama pull qwen3:8b

ChatOllama 默认连接 http://localhost:11434，无需任何 key。

参考文档：
  - ChatOllama 集成: https://python.langchain.com/docs/integrations/chat/ollama/
  - Ollama 模型库: https://ollama.com/search

安装：
  pip install langchain-ollama
"""

def demo_ollama():
    """
    使用 ChatOllama 调用本地模型
    参考: https://python.langchain.com/docs/integrations/chat/ollama/
    """
    from langchain_ollama import ChatOllama
    #TODO
    pass

if __name__ == "__main__":
    import subprocess
    # 检查 Ollama 是否在运行
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            capture_output=True, timeout=3
        )
        if result.returncode != 0:
            print("Ollama 未运行，请先启动：")
            print("  brew services start ollama")
            exit(1)
    except FileNotFoundError:
        print("curl 不可用，尝试直接连接...")
    except Exception as e:
        print(f"无法连接 Ollama: {e}")
        print("请确保 Ollama 正在运行")

    print("=== Ollama 本地模型 ===\n")
    demo_ollama()
