# 08_text_splitters - Text Splitters 文本切分器

Text Splitters 将长文档切分为小块（chunks），是 RAG 系统的关键步骤。

## 为什么需要切分？

- LLM 有上下文窗口限制，无法一次性处理整篇文档
- 切分后进行向量检索，只返回相关片段，提高准确性和效率
- 切块大小直接影响 RAG 质量：太大则检索不精确，太小则丢失上下文

## 教学路径

| 文件 | 内容 | 需要 Key |
|------|------|---------|
| `01_recursive_character.py` | RecursiveCharacterTextSplitter 全参数（首选默认） | ❌ |
| `02_character_and_markdown.py` | CharacterTextSplitter + MarkdownHeaderTextSplitter + MarkdownTextSplitter | ❌ |
| `03_splitter_summary.py` | 选型速查 + 参数调优指南 + 完整 RAG 切分流程 | ❌ |

## 三个切分方法

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# 方法 1: str -> list[str]
chunks = splitter.split_text("长文本...")

# 方法 2: list[str] -> list[Document]（附加 metadata）
docs = splitter.create_documents(["文本"], metadatas=[{"source": "a.txt"}])

# 方法 3: list[Document] -> list[Document]（保留原 metadata，**最常用**）
docs = splitter.split_documents(loader.load())
```

## Splitter 选型

| Splitter | 切分依据 | 返回类型 | 推荐场景 |
|----------|---------|---------|---------|
| **RecursiveCharacterTextSplitter** | 字符数 + 多分隔符递归 | str / Document | **通用首选** |
| CharacterTextSplitter | 单个分隔符 | str / Document | 结构清晰的文本 |
| MarkdownHeaderTextSplitter | Markdown 标题层级 | Document | 技术文档/Wiki |
| MarkdownTextSplitter | Markdown 结构 + chunk_size | str | 需控制大小的 Markdown |
| TokenTextSplitter | Token 数（需 tiktoken） | str / Document | 精确控制 token |
| PythonCodeTextSplitter | Python 语法 | str / Document | Python 源码 |

## RecursiveCharacterTextSplitter 默认分隔符

```
["\n\n", "\n", " ", ""]
```

递归顺序：段落 > 换行 > 空格 > 逐字符。优先在语义自然的位置切分。

## 参数调优建议

| 参数 | 建议值 | 说明 |
|------|-------|------|
| chunk_size | 300-1000 | 起始值 500 |
| chunk_overlap | chunk_size 的 10%-20% | 防止语义被切断 |
| separators | 中文加 "。" "，" | 提高中文切分质量 |
| length_function | 默认 len() | 可改为 token 计数 |

## 运行

```bash
# 全部无需 API Key
python 08_text_splitters/01_recursive_character.py
python 08_text_splitters/02_character_and_markdown.py
python 08_text_splitters/03_splitter_summary.py
```
