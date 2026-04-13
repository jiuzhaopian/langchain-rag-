# 09_retrieval - 检索（Retrieval）

LangChain 的 Retrieval 模块负责从外部数据源中检索与查询相关的文档，是 RAG（检索增强生成）的核心。

## 文件说明

| 文件 | 内容 | 依赖 |
|------|------|------|
| `01_vector_store.py` | InMemoryVectorStore 基础操作（创建、搜索、删除） | langchain-core |
| `02_retriever.py` | Retriever 体系（VectorStore、BM25、MultiQuery） | langchain-core, rank_bm25 |
| `03_rag_chain.py` | 完整 RAG 链（检索 + LLM 生成） | langchain, langchain-community, ZHIPUAI_API_KEY + Ollama |

## 核心概念

```
文档 → Embedding → VectorStore → Retriever → Chain → LLM
```

### VectorStore（向量存储）
存储文档的向量表示，支持相似度搜索。`InMemoryVectorStore` 适合学习，生产用 Chroma/FAISS。

### Retriever（检索器）
统一的检索接口，`invoke(query) → list[Document]`：
- **VectorStoreRetriever**: 基于向量相似度（dense retrieval），语义理解能力强
- **BM25Retriever**: 基于关键词匹配（sparse retrieval），不需要 Embedding，适合精确查询
- **MultiQueryRetriever**: 用 LLM 改写查询，提高召回率

### RAG Chain
将 Retriever 和 LLM 串联，让 LLM 基于检索到的文档生成回答。

## 运行

```bash
# 01、02 无需外部依赖，可直接运行
python 01_vector_store.py
python 02_retriever.py          # BM25 需要: pip install rank_bm25

# 03 需要 LLM 和 Embedding
ZHIPUAI_API_KEY=your_key python 03_rag_chain.py  # 还需要 Ollama bge-m3 服务
```

**注意**: 本模块在 langchain 1.2.15 中测试通过。`create_retrieval_chain` API 在旧版本中可能不可用，已展示 LCEL 替代方案。
