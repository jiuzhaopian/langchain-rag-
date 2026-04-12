# 03_embeddings - Embeddings 组件教学案例

Embeddings 将文本转为向量，是语义搜索和 RAG 的基础。

## 教学路径

| 文件 | 内容 | 需要 Key |
|------|------|---------|
| `01_ollama_embeddings.py` | 本地 bge-m3 + 余弦相似度原理 | ❌ |
| `02_dashscope_embeddings.py` | 通义 text-embedding-v3（原生端点 + OpenAI 兼容端点对比） | ✅ DASHSCOPE |
| `03_vectorstore_search.py` | VectorStore 语义搜索 + as_retriever 全参数 | ❌ |

## 余弦相似度公式

```
cos(θ) = (A · B) / (|A| × |B|)

A · B = A₁B₁ + A₂B₂ + ... + AₙBₙ    （向量点积）
|A|  = √(A₁² + A₂² + ... + Aₙ²)      （向量模长）

取值范围: [-1, 1]，1 = 最相似，0 = 无关，-1 = 最不相似
```

## VectorStore.as_retriever 常用参数

| 参数 | 说明 |
|------|------|
| `search_type="similarity"` | 默认，纯相似度搜索 |
| `search_type="mmr"` | 最大边际相关性，平衡相关性和多样性 |
| `search_type="similarity_score_threshold"` | 带相似度阈值过滤 |
| `k=2` | 返回 top-k 条结果 |
| `fetch_k=10` | mmr 先候选 fetch_k 条，再选 k 条 |
| `lambda_mult=0.5` | mmr 多样性参数，0=最大多样性，1=纯相似度 |
| `score_threshold=0.5` | 最低相似度阈值 |

## DashScope Embeddings 两种接入方式

| 方式 | 类 | 端点 | 包 |
|------|-----|------|-----|
| 原生（推荐） | `DashScopeEmbeddings` | DashScope 原生 API | `langchain-community` |
| OpenAI 兼容 | `OpenAIEmbeddings` | `/compatible-mode/v1/` | `langchain-openai` |

## 运行

```bash
# 本地（无需 Key）
python 03_embeddings/01_ollama_embeddings.py
python 03_embeddings/03_vectorstore_search.py

# 需要 DASHSCOPE_API_KEY
python 03_embeddings/02_dashscope_embeddings.py
```
