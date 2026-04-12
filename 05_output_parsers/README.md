# 05 Output Parsers - 输出解析器

将 LLM 的原始文本输出转换为结构化数据（dict、list、Pydantic 对象等）。

## 核心概念

Output Parser 是 LangChain 表达式语言（LCEL）链的最后一环：

```
prompt | llm | parser
              ^^^^^^ 输出 AIMessage → 输出结构化数据
```

**注意**：现代 LLM 大多支持原生 structured output，Output Parser 主要用于：
- 不支持原生结构化输出的模型
- 需要额外验证或后处理

## 文件列表

| 文件 | 内容 | 需要 Key |
|------|------|----------|
| `01_string_json.py` | StrOutputParser + JsonOutputParser：提取文本、解析 JSON、LCEL 链中 parser 的位置 | 智谱 |
| `02_pydantic_parser.py` | PydanticOutputParser：Pydantic 模型定义结构、嵌套模型、列表字段 | 智谱 |
| `03_list_parser.py` | CommaSeparatedListOutputParser + ListOutputParser：列表解析、三种 parser 对比 | 智谱 |
| `04_custom_parser.py` | BaseOutputParser 自定义：键值对解析、评分提取、带容错的 JSON 解析（无需 LLM） | 智谱（前两个 demo） |

## 常用 Parser 速查

| Parser | 输出类型 | 使用场景 |
|--------|----------|----------|
| `StrOutputParser` | `str` | 去掉 message 包装，提取纯文本 |
| `JsonOutputParser` | `dict` | 解析 JSON，不做类型验证 |
| `PydanticOutputParser` | Pydantic 对象 | 严格类型验证，推荐生产使用 |
| `CommaSeparatedListOutputParser` | `list[str]` | 枚举类场景（逗号分隔） |
| `ListOutputParser` | `list[str]` | 每行一个元素的列表 |
| 自定义（继承 `BaseOutputParser`） | 自定义 | 格式化输出、后处理、遗留系统 |

## 安装

```bash
pip install langchain-core langchain-community zhipuai pydantic
```
