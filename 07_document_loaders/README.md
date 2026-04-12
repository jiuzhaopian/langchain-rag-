# 07_document_loaders - Document Loaders 文档加载器

Document Loaders 将各种格式的外部数据加载为 LangChain 的 `Document` 对象。

## Document 数据结构

```python
from langchain_core.documents import Document

doc = Document(
    page_content="文本内容",        # str: 文本内容
    metadata={"source": "a.txt"},  # dict: 元数据（文件路径、页码等）
)
```

Document 贯穿整个 RAG 流程：`Loaders 加载 → Splitters 切分 → VectorStore 存储 → Retriever 检索`

## 教学路径

| 文件 | 内容 | 需要 Key |
|------|------|---------|
| `01_text_loader.py` | TextLoader 纯文本加载 + Document 结构详解 + lazy_load | ❌ |
| `02_directory_loader.py` | DirectoryLoader 目录批量加载 + glob 匹配 + silent_errors | ❌ |
| `03_csv_json_loader.py` | CSVLoader（每行一个 Document）+ JSONLoader（jq 表达式） | ❌ |
| `04_pdf_loader.py` | PyPDFLoader（每页一个 Document）+ 懒加载 | ❌ |
| `05_loader_summary.py` | 统一接口 + 常用 Loader 速查 + RAG 数据流全景 | ❌ |

## Loader 统一接口

```python
loader = TextLoader("./file.txt")

docs = loader.load()           # List[Document]，一次性加载
docs = list(loader.lazy_load())  # Iterator[Document]，逐条产出
```

## 常用 Loader 速查

| Loader | 来源 | 额外依赖 | 说明 |
|--------|------|---------|------|
| TextLoader | langchain-community | 无 | 纯文本文件 |
| DirectoryLoader | langchain-community | 无 | 目录批量（需指定 loader_cls） |
| CSVLoader | langchain-community | 无 | CSV（每行一个 Document） |
| JSONLoader | langchain-community | jq | JSON（支持 jq 表达式提取） |
| PyPDFLoader | langchain-community | pypdf | PDF（每页一个 Document） |
| WebBaseLoader | langchain-community | bs4 | 网页 |

## DirectoryLoader 注意事项

必须指定 `loader_cls`，否则默认使用 UnstructuredLoader（需安装 unstructured 包）：

```python
# 正确
DirectoryLoader("./docs", glob="**/*.txt", loader_cls=TextLoader)

# 错误（需要 pip install unstructured）
DirectoryLoader("./docs", glob="**/*.txt")  # 默认 UnstructuredLoader
```

## 运行

```bash
# 全部无需 API Key
python 07_document_loaders/01_text_loader.py
python 07_document_loaders/02_directory_loader.py
python 07_document_loaders/03_csv_json_loader.py
python 07_document_loaders/04_pdf_loader.py
python 07_document_loaders/05_loader_summary.py
```
