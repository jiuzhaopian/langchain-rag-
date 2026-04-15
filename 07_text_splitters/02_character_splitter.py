"""
02_character_splitter.py - CharacterTextSplitter

CharacterTextSplitter 按单个指定分隔符切分，简单直接。
与 RecursiveCharacterTextSplitter 的关键区别：
  - 只用一个分隔符，不会递归尝试其他分隔符
  - 如果某段文本超过 chunk_size，不会继续拆分（直接保留整段）
  - 适合已有天然分隔符的文本（如 JSON、日志等）

参考文档：
  - CharacterTextSplitter: https://reference.langchain.com/python/langchain-text-splitters/character/CharacterTextSplitter
  - 源码：https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/character.py

安装：
  pip install langchain-text-splitters
"""

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter


# ============================================================
# 演示 1：基本切分
# ============================================================

def demo_basic():
    """
    CharacterTextSplitter 按指定分隔符切分。
    切分后每个块如果超过 chunk_size，**不会继续拆分**，直接保留整段。
    """
    print("=== 演示 1：CharacterTextSplitter 基本切分 ===")

    # 三段文本，用 \\n\\n 分隔
    text = (
        "第一段：这是第一段内容，它的长度超过 chunk_size=50 的限制，"
        "但 CharacterTextSplitter 不会继续拆分。\n\n"
        "第二段：短内容。\n\n"
        "第三段：也是短内容。"
    )

    splitter = CharacterTextSplitter(
        separator="\n\n",    # 只用这一个分隔符
        chunk_size=50,       # 每块最大字符数
        chunk_overlap=0,     # 无重叠
    )
    chunks = splitter.split_text(text)

    print(f"分隔符='\\n\\n', chunk_size=50,切分为 {len(chunks)} 个块")
    for i, chunk in enumerate(chunks):
        print(f"\n  块 {i+1}（{len(chunk)} 字符）:{chunk}")

    print(f"---分析---")
    print(f"  块 1: {len(chunks[0])} 字符 > chunk_size=50")
    print(f"    → 超长，但 CharacterTextSplitter 只用 \\n\\n 这一个分隔符")
    print(f"    → 块内没有 \\n\\n 了，所以不会继续拆分，整段保留")
    print(f"  块 2: {len(chunks[1])} 字符")
    print(f"    → CharacterTextSplitter 也会合并相邻的小段（和 Recursive 类似）")
    print(f"    → 第二段 + 第三段合起来 < chunk_size，所以合并为一个块")
    print(f"\n  结论：CharacterTextSplitter 对超长块不递归拆分，但短块会合并")


# ============================================================
# 演示 2：与 RecursiveCharacterTextSplitter 对比
# ============================================================

def demo_vs_recursive():
    """
    同样一段超长文本，对比两种 Splitter 的行为差异。
    """
    print("\n\n=== 演示 2：CharacterTextSplitter vs RecursiveCharacterTextSplitter ===")

    text = (
        "第一段内容很长很长很长很长很长很长很长很长很长很长很长很长很长很长很长很长很长很长，远远超过 chunk_size。\n\n"
        "第二段短。\n\n"
        "第三段短。"
    )

    char_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=50, chunk_overlap=0)
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)

    char_chunks = char_splitter.split_text(text)
    recursive_chunks = recursive_splitter.split_text(text)

    print(f"原文: 3 段（\\n\\n 分隔），第一段超长\n")
    print(f"CharacterTextSplitter: {len(char_chunks)} 个块")
    for i, c in enumerate(char_chunks):
        print(f"  块 {i+1}（{len(c)} 字符）: {c}")

    print(f"\nRecursiveCharacterTextSplitter: {len(recursive_chunks)} 个块")
    for i, c in enumerate(recursive_chunks):
        print(f"  块 {i+1}（{len(c)} 字符）: {c}")

    print(f"\n差异:")
    print(f"  CharacterTextSplitter: 超长块不拆 → 块大小不可控")
    print(f"  RecursiveCharacterTextSplitter: 超长块继续递归拆 → 块大小可控")


if __name__ == "__main__":
    demo_basic()
    demo_vs_recursive()
