"""
01_ollama_embeddings.py
Ollama 本地 Embeddings - 无需 API Key，完全本地

前提条件：
  1. Ollama 运行中
  2. 已拉取向量模型: ollama pull bge-m3

Embeddings 将文本转换为固定维度的浮点数向量。
两个核心方法：
  - embed_query(text)       → List[float]          嵌入单条查询
  - embed_documents(texts)  → List[List[float]]     嵌入多条文档

参考文档：
  - Embedding model integrations: https://docs.langchain.com/oss/python/integrations/embeddings
  - Ollama Embeddings: https://docs.langchain.com/oss/python/integrations/embeddings/ollama

安装：
  pip install langchain-ollama numpy
"""

import numpy as np
from langchain_ollama import OllamaEmbeddings


def cosine_similarity(v1: list, v2: list) -> float:
    """
    余弦相似度（Cosine Similarity）

    公式: cos(θ) = (A · B) / (|A| × |B|)

    其中：
      A · B = A₁B₁ + A₂B₂ + ... + AₙBₙ   （向量点积）
      |A|  = √(A₁² + A₂² + ... + Aₙ²)      （向量模长/范数）

    取值范围: [-1, 1]
      1  → 两个向量方向完全一致（最相似）
      0  → 两个向量正交（无关）
      -1 → 两个向量方向相反（最不相似）

    为什么用余弦相似度而不是欧氏距离？
      - 余弦相似度只关注方向，不关注大小
      - 长文档和短文档即使长度不同，只要语义相近，余弦值就高
      - 这是 Embedding 相似度计算的标配方法
    """
    a, b = np.array(v1), np.array(v2)
    dot_product = np.dot(a, b)                          # 点积: A · B
    norm_a = np.linalg.norm(a)                           # 模长: |A|
    norm_b = np.linalg.norm(b)                           # 模长: |B|
    return dot_product / (norm_a * norm_b)               # cos(θ)


def euclidean_distance(v1: list, v2: list) -> float:
    """
    欧氏距离（Euclidean Distance）

    公式: d(A, B) = √(Σ(Aᵢ - Bᵢ)²)

    即向量各维度差值的平方和再开根号，就是几何意义上的"直线距离"。

    取值范围: [0, +∞)
      0  → 两个向量完全相同
      越大 → 差异越大

    为什么 Embedding 相似度更常用余弦相似度而非欧氏距离？
      - Embedding 向量通常维度很高（如 bge-m3 的 1024 维），欧氏距离会被维度放大
      - 不同长度的文本产生的向量模长不同，欧氏距离会受长度影响
      - 余弦相似度归一化了模长，只比较方向，更适合语义相似度场景
      - 欧氏距离适合需要考虑绝对差值的场景（如聚类中的 K-Means）
    """
    a, b = np.array(v1), np.array(v2)
    return np.sqrt(np.sum((a - b) ** 2))               # √(Σ(Aᵢ - Bᵢ)²)


def demo_ollama_embeddings():
    """
    使用 OllamaEmbeddings（本地向量模型）
    """
    # TODO
    pass



def demo_similarity():
    """
    语义相似度计算 - 用余弦相似度衡量两段文本的语义接近程度
    """

    embeddings = OllamaEmbeddings(model="bge-m3")

    print("\n=== 相似度演示 ===\n")
    print("原理: 将文本转为向量，计算两个向量的夹角余弦值")
    print("值越接近 1 表示语义越相似\n")

    pairs = [
        ("我喜欢编程", "我热爱写代码"),     # 语义相近
        ("我喜欢编程", "今天天气真好"),      # 语义无关
        ("今天天气真好", "阳光明媚"),        # 语义相近
    ]
    for a, b in pairs:
        va = embeddings.embed_query(a)
        vb = embeddings.embed_query(b)
        sim = cosine_similarity(va, vb)
        dist = euclidean_distance(va, vb)
        bar = "█" * int(sim * 20)
        print(f"  '{a}' vs '{b}'")
        print(f"  余弦相似度: {sim:.4f}  {bar}")
        print(f"  欧氏距离:   {dist:.4f}")
        print()

    print("--- 观察 ---")
    print("余弦相似度越高 → 欧氏距离越小（方向一致 + 距离近）")
    print("但欧氏距离还受向量模长影响，不如余弦相似度稳定")


if __name__ == "__main__":
    import subprocess
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            capture_output=True, timeout=3
        )
        if result.returncode != 0:
            print("Ollama 未运行，请先启动")
            exit(1)
    except Exception as e:
        print(f"无法连接 Ollama: {e}")
        exit(1)

    demo_ollama_embeddings()
    demo_similarity()
