"""
02_dashscope_embeddings.py
通义千问 DashScope Embeddings - 两种接入方式

DashScope Embeddings 有两种接入方式：

方式 1: DashScopeEmbeddings（原生端点）
  - 使用 langchain-community 的 DashScopeEmbeddings 类
  - 直接调用阿里云 DashScope 原生 API
  - 安装: pip install dashscope langchain-community

方式 2: OpenAIEmbeddings（OpenAI 兼容端点）
  - 使用 langchain-openai 的 OpenAIEmbeddings 类
  - 走 DashScope 的 OpenAI 兼容接口
  - 安装: pip install langchain-openai

推荐方式 1（原生端点），与 dashscope chat models 保持一致。

参考文档：
  - DashScope 官方: https://help.aliyun.com/zh/model-studio/use-bailian-in-langchain
  - DashScope Embeddings: https://docs.langchain.com/oss/python/integrations/embeddings/dashscope

安装：
  pip install dashscope langchain-community langchain-openai
"""

import os

import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings

# ============================================================
# 公共：余弦相似度
# ============================================================

def cosine_similarity(v1: list, v2: list) -> float:
    """
    余弦相似度（Cosine Similarity）

    公式: cos(θ) = (A · B) / (|A| × |B|)

    取值范围: [-1, 1]，1 = 最相似，0 = 无关
    详见 01_ollama_embeddings.py 中的完整说明
    """
    a, b = np.array(v1), np.array(v2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ============================================================
# 公共：Embeddings 实例
# ============================================================

def get_dashscope_embeddings():
    """方式 1：DashScope 原生端点（推荐）"""
    #TODO
    pass


def get_openai_compatible_embeddings():
    """方式 2：OpenAI 兼容端点"""
    #TODO
    pass


# ============================================================
# 演示 1：DashScope 原生端点
# ============================================================

def demo_dashscope_native():
    """
    方式 1：DashScopeEmbeddings（原生端点）
    """
    embeddings = get_dashscope_embeddings()

    print("=== DashScope Embeddings（原生端点） ===")

    # embed_query - 嵌入单条查询
    query_vector = embeddings.embed_query("什么是机器学习？")
    print(f"查询向量维度: {len(query_vector)}")
    print(f"前 5 维: {query_vector[:5]}")

    # embed_documents - 嵌入多条文档
    documents = [
        "LangChain 是一个用于构建 LLM 应用的框架",
        "Python 是一门流行的编程语言",
    ]
    doc_vectors = embeddings.embed_documents(documents)
    print(f"文档向量数量: {len(doc_vectors)}")
    print(f"每个向量维度: {len(doc_vectors[0])}")


# ============================================================
# 演示 2：OpenAI 兼容端点
# ============================================================

def demo_openai_compatible():
    """
    方式 2：OpenAIEmbeddings（OpenAI 兼容端点）

    如果你已有 OpenAI 兼容的代码，只需改 base_url 和 api_key
    就能切换到 DashScope，无需改其他代码。
    """
    embeddings = get_openai_compatible_embeddings()

    print("\n=== DashScope Embeddings（OpenAI 兼容端点） ===")

    query_vector = embeddings.embed_query("什么是机器学习？")
    print(f"查询向量维度: {len(query_vector)}")
    print(f"前 5 维: {query_vector[:5]}")


# ============================================================
# 演示 3：两种方式对比
# ============================================================

def demo_comparison():
    """
    两种方式结果一致（同一个模型、同一个端点），只是接入方式不同。
    """
    e1 = get_dashscope_embeddings()
    e2 = get_openai_compatible_embeddings()

    text = "LangChain 是什么？"
    v1 = e1.embed_query(text)
    v2 = e2.embed_query(text)

    # 余弦相似度（应该接近 1.0）
    sim = cosine_similarity(v1,v2)

    print("\n=== 两种方式对比 ===")
    print(f"  原生端点维度: {len(v1)}")
    print(f"  兼容端点维度: {len(v2)}")
    print(f"  余弦相似度:   {sim:.6f}  {'✅ 一致' if sim > 0.99 else '⚠️ 有差异'}")


if __name__ == "__main__":
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("请设置 DASHSCOPE_API_KEY 环境变量")
        exit(1)

    demo_dashscope_native()
    demo_openai_compatible()
    demo_comparison()
