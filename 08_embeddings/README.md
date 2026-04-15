# 08_embeddings - Embeddings 组件教学案例

Embeddings 将文本转为向量，是语义搜索和 RAG 的基础。

## 教学路径

| 文件 | 内容 | 需要 Key |
|------|------|---------|
| `01_ollama_embeddings.py` | 本地 bge-m3 + 余弦相似度原理 | ❌ |
| `02_dashscope_embeddings.py` | 通义 text-embedding-v3（原生端点 + OpenAI 兼容端点对比） | ✅ DASHSCOPE |

## 余弦相似度公式

```
cos(θ) = (A · B) / (|A| × |B|)

A · B = A₁B₁ + A₂B₂ + ... + AₙBₙ    （向量点积）
|A|  = √(A₁² + A₂² + ... + Aₙ²)      （向量模长）

取值范围: [-1, 1]，1 = 最相似，0 = 无关，-1 = 最不相似
```

## DashScope Embeddings 两种接入方式

| 方式 | 类 | 端点 | 包 |
|------|-----|------|-----|
| 原生（推荐） | `DashScopeEmbeddings` | DashScope 原生 API | `langchain-community` |
| OpenAI 兼容 | `OpenAIEmbeddings` | `/compatible-mode/v1/` | `langchain-openai` |

## 运行

```bash
# 本地（无需 Key）
python 08_embeddings/01_ollama_embeddings.py

# 需要 DASHSCOPE_API_KEY
python 08_embeddings/02_dashscope_embeddings.py
```
