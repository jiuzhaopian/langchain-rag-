# LangChain 学习指南

## 第一章：简介

LangChain 是一个用于构建 **LLM 应用**的开发框架。

核心组件包括：

- **Models**：与大语言模型交互的接口
- **Prompts**：提示词管理
- **Chains**：链式调用
- **Retrievers**：检索增强

## 第二章：快速开始

安装：

```python
pip install langchain
```

基础使用：

```python
from langchain_core.messages import HumanMessage
```

> 提示：建议使用虚拟环境管理依赖。

## 第三章：进阶主题

1. RAG（检索增强生成）
2. Agent（自主决策）
3. Memory（对话记忆）

详见 [官方文档](https://python.langchain.com)。
