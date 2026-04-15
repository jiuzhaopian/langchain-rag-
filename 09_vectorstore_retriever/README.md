# 09_vectorstore_retriever - VectorStore + Retriever 教学案例

VectorStore 存储和检索向量化的文档，Retriever 是 LangChain 的标准检索接口。

## 教学路径

| 文件 | 内容 | 需要 Key | 额外依赖 |
|------|------|---------|---------|
| `01_inmemory_vectorstore.py` | InMemoryVectorStore 基础：创建/搜索/带分数/阈值过滤/删除 | ❌ | langchain-core, langchain-ollama |
| `02_faiss_vectorstore.py` | FAISS：高性能搜索/merge增量/persistence/一致性对比 | ❌ | faiss-cpu, langchain-community |
| `03_chroma_vectorstore.py` | Chroma：metadata过滤/持久化/增量添加删除/三者对比 | ❌ | langchain-chroma |
| `04_retriever.py` | as_retriever + similarity/MMM/score_threshold + LCEL集成 | ❌ | faiss-cpu (demo4) |

## 三种 VectorStore 对比

| 特性 | InMemory | FAISS | Chroma |
|------|----------|-------|--------|
| 持久化 | ❌ | save_local 手动 | ✅ 自动 |
| 元数据过滤 | ❌ | ❌ | ✅ 原生 |
| 增量添加 | ✅ add_texts | merge_from | ✅ add_texts |
| 删除 | delete(ids) | ❌ | ✅ where条件 |
| 大数据性能 | 慢 | ⚡ 快（亿级） | 中等（百万级） |
| 适用场景 | 教学/测试 | 生产/高性能 | 开发/中小规模 |

## Retriever 搜索模式

| search_type | 说明 | 关键参数 |
|------------|------|---------|
| similarity | 默认，纯相似度 Top-K | k |
| mmr | 平衡相关性和多样性 | k, fetch_k, lambda_mult |
| similarity_score_threshold | 带阈值过滤 | k, score_threshold |

## 运行

```bash
# 前提：Ollama 运行中，已拉取 bge-m3

# 基础（无额外依赖）
python 09_vectorstore_retriever/01_inmemory_vectorstore.py

# 需要 faiss-cpu
pip install faiss-cpu langchain-community
python 09_vectorstore_retriever/02_faiss_vectorstore.py

# 需要 langchain-chroma
pip install langchain-chroma
python 09_vectorstore_retriever/03_chroma_vectorstore.py

# 需要 faiss-cpu（demo4 使用 FAISS 演示阈值过滤）
python 09_vectorstore_retriever/04_retriever.py
```
