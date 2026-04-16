# 10_rag_chain - 完整 RAG 应用

综合前面所有模块的知识点（Chat Models + Messages + Prompts + Output Parsers + Chains + Document Loaders + Text Splitters + Embeddings + VectorStore），构建一个完整的 RAG 对话应用。

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI (02)                       │
│  ┌─────────────────┐  ┌───────────────────────────────┐  │
│  │  📄 文档管理     │  │  💬 RAG 对话                  │  │
│  │  - 上传/切分/清空 │  │  - 多轮对话                    │  │
│  │  - chunk调优      │  │  - 流式/非流式切换              │  │
│  └─────────────────┘  │  - 检索调优(k/MMR/threshold)   │  │
│          │           └───────────────────────────────┘  │
│          ▼                      ▼                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              RAG Engine (01)                        │   │
│  │  ┌─────────────────────┐  ┌─────────────────────┐  │   │
│  │  │  DocumentManager    │  │    RAGEngine         │  │   │
│  │  │  - 加载/切分/存储    │  │  - chat()           │  │   │
│  │  │  - metadata管理     │  │  - chat_stream()    │  │   │
│  │  │  - retriever获取    │  │  - chain构建        │  │   │
│  │  └─────────────────────┘  └─────────────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│          │                       │                         │
│          ▼                       ▼                         │
│  ┌─────────────────┐  ┌───────────────────────────────┐  │
│  │ Chroma 向量库    │  │     LLM + Embedding          │  │
│  │ (持久化本地存储) │  │ 智谱 glm-4.7 + Ollama bge-m3 │  │
│  └─────────────────┘  └───────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                     │
                     ▼
           ┌─────────────────┐
           │ Config (00)    │
           │ - cosine距离    │
           │ - 历史轮数配置  │
           │ - 默认参数      │
           └─────────────────┘
```

## 文件结构

```
10_rag_chain/
├── 00_rag_config.py        # 全局配置：模型、向量库、切分、检索参数
├── 01_rag_engine.py         # 核心引擎：文档管理 + RAG 对话
├── 02_streamlit_app.py      # Streamlit UI：文档管理页 + RAG 对话页
├── data/chroma_db/          # Chroma 持久化目录（.gitignore 排除）
└── README.md
```

## 技术要点

### 1. 配置化设计 (00_rag_config.py)

- **Chroma 距离配置**：`CHROMA_COLLECTION_METADATA = {"hnsw:space": "cosine"}`
  - bge-m3 输出归一化向量，L2/cosine 排序等价但分数不同
  - cosine 分数 = 余弦相似度（直观），L2 分数被非线性压缩
- **历史轮数配置**：`DEFAULT_MAX_HISTORY_ROUNDS = 50`
  - 支持动态调整，避免 prompt 超长

### 2. 引擎与 UI 分离 (01_rag_engine.py)

- **DocumentManager**：文档加载、切分、存储
  - 支持格式：txt / md / csv / pdf
  - 自动附加 `source_file` metadata
  - 支持增量添加和清空重建

- **RAGEngine**：检索 + 生成 + 多轮对话
  - **chat()**：教学演示风格
    - 手动调用 `retriever.invoke()`
    - 手动拼装 context 和历史
    - chain 只有 3 个节点：`prompt | llm | parser`
  - **chat_stream()**：实际开发风格
    - 全流程 chain：`RunnableParallel → 拼装 → prompt → llm → parser`
    - 支持流式输出（逐 token yield）

### 3. Streamlit UI (02_streamlit_app.py)

- **两个页面**：
  - 📄 文档管理：上传文件、切分参数、查看已上传文档、清空向量库
  - 💬 RAG 对话：多轮对话、流式/非流式切换、侧边栏检索调优

- **检索调优参数**：
  - `k`：返回文档数（1~10）
  - `search_type`：similarity / mmr / similarity_score_threshold
  - `score_threshold`：相似度阈值（0~1）
  - `mmr_lambda`：MMR 多样性参数（0~1）

### 4. ChatPromptTemplate 消息类型 (02 Messages)

```python
self._prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="系统角色定义"),
    ("system", "检索到的文档：\n{context}"),
    MessagesPlaceholder("history"),  # 历史消息动态插入
    ("human", "{question}"),
])
```

### 5. LCEL Chain 两种写法 (06 Chains)

**写法 1：手动控制（chat()）**

```python
# 1. 手动检索
docs = retriever.invoke(query)
context = "\n\n".join(f"[来源: {d.metadata['source_file']}]\n{d.page_content}" for d in docs)

# 2. 手动构建历史
history_messages = [HumanMessage(c), AIMessage(c), ...]

# 3. 3节点 chain
chain = self._prompt | self.llm | self._parser
response = chain.invoke({"context": context, "history": history_messages, "question": query})
```

**写法 2：全流程 chain（chat_stream()）**

```python
# RunnableParallel + RunnablePassthrough 组合
chain = (
    RunnableParallel(
        context=retriever | format_docs,  # 检索并格式化
        question=RunnablePassthrough(),    # 透传 query
    )
    | (lambda inputs: {"context": inputs["context"], "history": history, "question": inputs["question"]})
    | self._prompt | self.llm | self._parser
)

# 流式输出
for token in chain.stream(query):
    yield token
```

### 6. Chroma Cosine 距离 (09 VectorStore)

```python
CHROMA_COLLECTION_METADATA = {"hnsw:space": "cosine"}
```

**为什么显式指定 cosine？**

- 归一化向量下 L2 和 cosine 排序等价（L2² = 2(1-cos)）
- 但分数值不同：
  - cosine 下：分数 = 余弦相似度（如 0.6），直观易懂
  - L2 下：经公式转换后分数被压缩（如 0.37），"相关文档才 0.37" 容易困惑

## 使用方式

### 启动 Streamlit

```bash
cd 10_rag_chain
streamlit run 02_streamlit_app.py
```

### 独立使用引擎（不依赖 Streamlit）

```python
from 00_rag_config import get_llm, get_embeddings, CHROMA_PERSIST_DIR
from 01_rag_engine import DocumentManager, RAGEngine

# 初始化
embeddings = get_embeddings()
llm = get_llm()
doc_manager = DocumentManager(persist_dir=CHROMA_PERSIST_DIR, embeddings=embeddings)
rag_engine = RAGEngine(doc_manager=doc_manager, llm=llm)

# 上传文档
doc_manager.process_file("example.txt")

# 对话（非流式）
response = rag_engine.chat(query="LangChain 是什么？", history=[])

# 对话（流式）
for token in rag_engine.chat_stream(query="LangChain 是什么？", history=[]):
    print(token, end="")
```

## 知识点映射

| 模块 | 知识点 | 代码位置 |
|------|--------|----------|
| 01_chat_models | Chat Models | `get_llm()` |
| 02_messages | Message 类型 | `_build_history_messages()` |
| 03_prompts | ChatPromptTemplate | `RAGEngine.__init__()` |
| 03_prompts | MessagesPlaceholder | `self._prompt` |
| 04_output_parsers | StrOutputParser | `self._parser` |
| 06_chains | RunnablePassthrough | `chat_stream()` |
| 06_chains | RunnableParallel | `chat_stream()` |
| 06_document_loaders | Document Loaders | `DocumentManager.LOADER_MAP` |
| 07_text_splitters | RecursiveCharacterTextSplitter | `DocumentManager._splitter` |
| 08_embeddings | Ollama Embeddings | `get_embeddings()` |
| 09_vectorstore_retriever | Chroma | `DocumentManager.vectorstore` |
| 09_vectorstore_retriever | as_retriever | `DocumentManager.get_retriever()` |

## 依赖

```txt
streamlit>=1.30.0
langchain>=0.3.0
langchain-community>=0.3.0
langchain-chroma>=0.1.0
langchain-text-splitters>=0.3.0
langchain-zhipu>=0.3.0
pypdf>=4.0.0
ollama>=0.4.0
```

## 参考资料

- Streamlit 文档: https://docs.streamlit.io/
- LangChain RAG: https://python.langchain.com/docs/tutorials/rag/
- ChatPromptTemplate: https://python.langchain.com/docs/concepts/#prompt-templates
- RunnableParallel: https://python.langchain.com/docs/concepts/#runnableparallel
