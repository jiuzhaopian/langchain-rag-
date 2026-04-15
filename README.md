# LangChain Study - LangChain 1.2.x 教学案例

基于 LangChain 1.2.x 官方文档，使用国内可用 API（通义/DeepSeek/智谱/Ollama）。

## 重要说明

LangChain 1.x 已将旧版 LLMs（纯文本补全）和 Chat Models 合并为**统一的 Models 接口**。
新项目直接使用 Chat Models 即可，旧版 LLM 仅作了解。

### 旧版 LLM vs 新版 Chat Model

| 特性 | 旧版 LLM (BaseLLM) | 新版 Chat Model (BaseChatModel) |
|------|---------------------|-------------------------------|
| 输入 | 纯字符串 `str` | 消息列表 `List[BaseMessage]` |
| 输出 | 纯字符串 `str` | `AIMessage` 对象 |
| 典型类 | `OpenAI` (gpt-3.5-turbo-instruct) | `ChatOpenAI` (gpt-4o, qwen-plus) |
| 状态 | **已废弃** | **LangChain 1.x 唯一推荐** |

**为什么废弃？**

1. 所有主流模型厂商（OpenAI、通义、智谱、DeepSeek）都已转向 Chat API，纯补全 API 已停止维护
2. Chat Model 功能是 LLM 的超集：支持多轮对话、系统提示、工具调用，纯补全做不到
3. 即使只做单轮文本生成，Chat Model 也能直接接收字符串（自动包装为 HumanMessage），使用成本相同
4. LangChain 1.x 的 `init_chat_model()` 只支持 Chat Model，不再支持旧版 LLM

**结论：新项目不要用旧版 LLM，直接用 Chat Models。**

## 教学路径

```
01_chat_models/        ← Chat Models 全用法
├── 01_openai_compatible.py    通过 ChatOpenAI + base_url 兼容 OpenAI 协议的服务
├── 02_dashscope_native.py     通义千问 DashScope 原生接口（ChatTongyi）
├── 03_deepseek.py             DeepSeek（OpenAI 兼容）
├── 04_zhipu.py                智谱 GLM（官方 langchain-glm 包）
├── 05_ollama.py               本地模型 Ollama
├── 06_init_chat_model.py      通用初始化器 init_chat_model
├── 07_stream.py               流式调用
├── 08_batch.py                批量调用
└── README.md

02_messages/           ← Messages 组件
├── 01_message_types.py        Message 类型：System/Human/AI/Tool
├── 02_message_formats.py      Message 写法对比
└── README.md

03_prompts/            ← Prompts 提示词模板
├── 01_prompt_template.py      PromptTemplate 纯文本模板
├── 02_chat_prompt_template.py ChatPromptTemplate 聊天模板（重点）
├── 03_messages_placeholder.py MessagesPlaceholder 对话历史插入
├── 04_few_shot_prompt.py      Few-Shot 少样本提示
└── README.md

04_output_parsers/     ← Output Parsers 输出解析器
├── 01_string_json.py         StrOutputParser + JsonOutputParser
├── 02_pydantic_parser.py     PydanticOutputParser（重点）
├── 03_list_parser.py         CommaSeparatedListOutputParser + ListOutputParser
├── 04_custom_parser.py       BaseOutputParser 自定义解析器
└── README.md

05_chains/             ← LCEL 链式调用（Chains）
├── 01_basic_chain.py          基本链 + @chain 装饰器
├── 02_parallel_branch.py      RunnableParallel + RunnableBranch
├── 03_passthrough.py          RunnablePassthrough/Pick + RunnableLambda
├── 04_chain_principle.py      LCEL 链原理剖析
└── README.md

06_document_loaders/   ← Document Loaders 文档加载
├── 01_text_loader.py          TextLoader 基础
├── 02_directory_loader.py     DirectoryLoader 目录加载
├── 03_csv_loader.py           CSV 文件加载
├── 04_json_loader.py          JSON 文件加载（jq 筛选）
├── 05_pdf_loader.py           PDF 文件加载
├── 06_unstructured_md_loader.py Markdown 加载
├── 07_loader_summary.py       总结 + 选型指南
└── README.md

07_text_splitters/     ← Text Splitters 文本分割
├── 01_recursive_character.py  RecursiveCharacterTextSplitter（通用首选）
├── 02_character_splitter.py   CharacterTextSplitter（单分隔符）
├── 03_markdown_splitter.py    MarkdownHeaderTextSplitter + MarkdownTextSplitter
├── 04_html_splitter.py        HTMLHeaderTextSplitter + HTMLSectionSplitter + HTMLSemanticPreservingSplitter
├── 05_latex_splitter.py       LatexTextSplitter
├── 06_splitter_summary.py     总结 + 选型速查 + 参数调优
└── README.md

08_embeddings/         ← Embeddings 向量模型（RAG 前置）
├── 01_ollama_embeddings.py    本地 bge-m3 + 余弦相似度/欧氏距离原理
├── 02_dashscope_embeddings.py 通义 text-embedding-v3（原生 + OpenAI 兼容对比）
└── README.md

09_vectorstore_retriever/ ← VectorStore + Retriever
├── 01_inmemory_vectorstore.py    InMemoryVectorStore 基础：创建/搜索/带分数/阈值过滤/删除
├── 02_faiss_vectorstore.py       FAISS：高性能/merge增量/persistence/一致性对比
├── 03_chroma_vectorstore.py      Chroma：metadata过滤/持久化/增量添加删除/三者对比
├── 04_retriever.py               as_retriever + similarity/MMR/score_threshold + LCEL集成
└── README.md
```

## 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 设置 API Key（至少设置一个）
export DASHSCOPE_API_KEY="sk-..."       # 通义千问
export DEEPSEEK_API_KEY="sk-..."        # DeepSeek
export ZHIPUAI_API_KEY="..."            # 智谱

# Ollama 需要本地运行（无 key）
ollama pull qwen3:8b                    # 拉取本地模型

# 运行案例
python 01_chat_models/01_openai_compatible.py
```

## API Key 来源

- 通义千问: https://dashscope.console.aliyun.com/apiKey
- DeepSeek: https://platform.deepseek.com/api_keys
- 智谱: https://open.bigmodel.cn/usercenter/apikeys
- Ollama: 本地运行，无需 key

## 参考

- LangChain Models 文档: https://docs.langchain.com/oss/python/langchain/models
- LangChain Retrieval: https://docs.langchain.com/oss/python/langchain/retrievers
- LangChain VectorStore: https://docs.langchain.com/oss/python/langchain/vectorstores
