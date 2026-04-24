"""
01_recursive_character.py - RecursiveCharacterTextSplitter（最常用）

RecursiveCharacterTextSplitter 是 LangChain 最常用的文本切分器。
它按字符数切分，递归尝试多种分隔符，优先在自然断点（段落 > 句子 > 单词）处切分。

原理：
  1. 先尝试用 "\\n\\n"（段落）切分，得到多个小段
  2. 将相邻小段**合并**，直到接近 chunk_size（不是切完就结束）
  3. 如果合并后的块仍超过 chunk_size，用 "\\n"（换行）继续拆
  4. 还太大用 " "（空格）切，最后逐字符切

先按最优先分隔符（如 \n\n）切分文本 → 遍历每个小段，不断将当前小段与下一个合并，直到总长度接近 chunk_size（考虑 overlap）→ 一旦超过 chunk_size，就回退一步，用更低一级的分隔符（如 \n）继续切分当前块
核心：先切后合并，保证块尽量大但不超 chunk_size，同时在语义自然的位置切分。

示例：SAMPLE_TEXT 有 6 个段落（\\n\\n 分隔），chunk_size=100 时：
  - 段落 1+2 合并 ≈ 96 字符 < 100 → 成为一个 chunk
  - 段落 3+4 合并 ≈ 95 字符 < 100 → 成为一个 chunk
  - 段落 5+6 合并 ≈ 70 字符 < 100 → 成为一个 chunk
  - 最终 3 个 chunk（不是 6 个）

参考文档：
  - RecursiveCharacterTextSplitter: https://reference.langchain.com/python/langchain-text-splitters/character/RecursiveCharacterTextSplitter
  - 源码：https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/character.py

安装：
  pip install langchain-text-splitters
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ============================================================
# 公共：示例文本
# ============================================================

SAMPLE_TEXT = """LangChain 是一个用于构建 LLM 应用的框架。

它提供了多种工具和抽象，包括 Models（模型）、Prompts（提示词）、Chains（链）和 Retrievers（检索器）。

开发者可以使用 LangChain 快速构建 RAG 系统、智能代理和各种 AI 应用。

RAG（Retrieval Augmented Generation）是一种结合检索和生成的技术。

它通过检索相关文档来增强 LLM 的回答质量，减少幻觉问题。

在实际应用中，RAG 系统通常包含文档加载、文本切分、向量存储和检索等步骤。"""


# ============================================================
# 演示 1：基本切分 - split_text
# ============================================================

def demo_split_text():
    """
    split_text: str -> list[str]（纯文本切分，不保留 metadata）
    """
    print("=== 演示 1：split_text（纯文本切分） ===")

    # TODO
    pass


# ============================================================
# 演示 2：带 metadata 切分 - create_documents
# ============================================================

def demo_create_documents():
    """
    create_documents: list[str] -> list[Document]（为每个块附加 metadata）
    """
    print("\n\n=== 演示 2：create_documents（带 metadata） ===")

    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)

    # TODO
    pass


# ============================================================
# 演示 3：切分已有 Document - split_documents
# ============================================================

def demo_split_documents():
    """
    split_documents: list[Document] -> list[Document]（保留原 metadata）
    最常用：从 Loader 加载的 Document 直接切分，metadata 原封不动传递。
    """
    print("\n\n=== 演示 3：split_documents（保留 metadata） ===")

    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)

    # 模拟从 Loader 加载的 Document
    # TODO
    pass



# ============================================================
# 演示 4：chunk_overlap 的作用与生效场景
# ============================================================

def demo_overlap():
    """
    chunk_overlap 的作用：
      当文本太长、没有现成的分隔符（如段落分隔）时，切分器会「硬切分」
      ——在句子甚至词语中间断开。这时 overlap 就像保险带，把切开位置的
      上下文各保留一些字，让相邻两个 chunk 在边界处有重叠，避免重要信息
      正好被切断在两个块之间。

    什么时候生效：
      文本中有大段连续内容（没有 \n\n 段落分隔），单段长度超过 chunk_size，
      切分器被迫用空格或逐字符来切时，overlap 才会产生实际重叠。

    什么时候不生效：
      文本有清晰的段落分隔，每段长度 < chunk_size，切分器总是能找到
      自然断点（段落/换行）来切分，不存在「硬切分」，overlap 也就无处施展。
      前面 demo 1-3 用的 SAMPLE_TEXT 就是这种情况（6 个短段落，
      每两段合并 ≈ 96 字符 < chunk_size=100），所以前面都没写 overlap。
    """
    print("\n\n=== 演示 4：chunk_overlap 的作用与生效场景 ===")

    # 一段没有换行的长文本——模拟现实中「一段话特别长」的场景
    # 单段 ≈170 字符，chunk_size=80，必然触发硬切分
    LONG_TEXT = (
        "在自然语言处理领域，Transformer 架构通过自注意力机制实现了对文本"
        "全局依赖关系的建模，彻底改变了传统的序列处理方式。与 RNN 不同，"
        "Transformer 可以并行处理整个序列，大幅提升了训练效率。BERT 模型"
        "基于 Transformer 的编码器部分，通过掩码语言模型预训练任务学习"
        "深层语义表示，在问答、分类等下游任务中表现优异。"
    )

    for overlap, label in [(20, "有重叠（推荐）"), (0, "无重叠")]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=overlap)
        chunks = splitter.split_text(LONG_TEXT)
        print(f"\n--- chunk_size=80, overlap={overlap}（{label}） ---")
        print(f"共 {len(chunks)} 个块:\n")
        for i, chunk in enumerate(chunks):
            print(f"  块{i+1}（{len(chunk)}字符）: {chunk}")
            # 标注与前一块的重叠
            if i > 0:
                prev = chunks[i - 1]
                for j in range(min(len(chunk), len(prev)), 0, -1):
                    if prev.endswith(chunk[:j]):
                        print(f"    ↑ 与前一块重叠 {j} 字符")
                        break

    print("\n--- 对比总结 ---")
    print("  overlap=0 :  块与块之间无缝连接，边界处可能正好把一句话切断")
    print("  overlap=20: 每个块开头会重复前一个块末尾的内容，上下文不丢失")

# ============================================================
# 演示 5：自定义分隔符
# ============================================================

def demo_separators():
    """
    默认分隔符: ["\n\n", "\n", " ", ""]
    可以自定义，如中文场景可以用中文句号、逗号作为分隔符。
    """
    print("\n\n=== 演示 5：自定义分隔符 ===")

    print("默认分隔符（递归顺序）: ['\\n\\n', '\\n', ' ', '']")
    print("  先尝试段落切分 -> 再换行 -> 再空格 -> 最后逐字符")

    # 中文场景自定义
    splitter_cn = RecursiveCharacterTextSplitter(
        chunk_size=80,
        chunk_overlap=10,
        separators=["\n\n", "。", "\n", "，", " ", ""],
    )
    chunks = splitter_cn.split_text(SAMPLE_TEXT)
    print(f"\n中文分隔符 ['\\n\\n', '。', '\\n', '，', ' ', '']:")
    print(f"  -> {len(chunks)} 个块")
    for i, chunk in enumerate(chunks):
        print(f"  块 {i+1};大小:{len(chunk)}: {chunk}")


# ============================================================
# 演示 6：length_function
# ============================================================

def demo_length_function():
    """
    length_function 控制如何计算"长度"。
    默认用 len()（字符数），可以改为 token 数等。
    """
    print("\n\n=== 演示 6：length_function ===")

    print("默认: length_function=len（按字符数计算）")
    print("可自定义: length_function=lambda x: len(tokenizer.encode(x))（按 token 数）")
    print("注意: 改 length_function 时，chunk_size 的含义也随之变化")

    # 示例：用自定义函数（如计算中文字符为 2 倍权重）
    def cn_weighted_len(text):
        """中文字符算 2，英文算 1"""
        return sum(2 if ord(c) > 127 else 1 for c in text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=cn_weighted_len,
    )
    chunks = splitter.split_text(SAMPLE_TEXT)
    print(f"\n中文加权长度（中文=2, 英文=1）:")
    print(f"  -> {len(chunks)} 个块")
    for i, chunk in enumerate(chunks):
        print(f"  块 {i + 1};大小:{len(chunk)}: {chunk}")


if __name__ == "__main__":
    demo_split_text()
    demo_create_documents()
    demo_split_documents()
    demo_overlap()
    demo_separators()
    demo_length_function()
