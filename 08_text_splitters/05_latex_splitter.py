"""
05_latex_splitter.py - LatexTextSplitter

LatexTextSplitter: 按 LaTeX 语义边界（章节、环境）切分，
同时遵守 chunk_size 限制。

参考文档：
  - LatexTextSplitter: https://reference.langchain.com/python/langchain-text-splitters/latex/LatexTextSplitter
  - 源码：https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/latex.py

安装：
  pip install langchain-text-splitters
"""

from langchain_text_splitters import LatexTextSplitter


def demo_latex_splitter():
    """
    LatexTextSplitter 按 LaTeX 文档结构切分：
    - 章节（\\section, \\subsection, \\subsubsection）
    - 环境（\\begin{...} ... \\end{...}）
    - 列表、表格等语义边界

    特点：
    - 返回 list[str]（不是 Document，没有 metadata）
    - 有 chunk_size 限制，控制块大小
    - 保持 LaTeX 环境的完整性（不会在 \\begin/\\end 中间断开）

    适合：学术论文、技术报告等 LaTeX 格式文档。
    """
    print("=== LatexTextSplitter ===")

    latex_text = r"""
\documentclass{article}
\begin{document}

\section{Introduction}
LangChain is a framework for building LLM applications.

\section{Core Components}
\subsection{Models}
Chat Models are the recommended model interface in LangChain 1.x.

\subsection{Prompts}
PromptTemplate and ChatPromptTemplate are used to manage prompts.

\begin{itemize}
\item PromptTemplate: for string prompts
\item ChatPromptTemplate: for chat message prompts
\end{itemize}

\section{RAG System}
RAG = Document Loading + Text Splitting + Vector Storage + Retrieval + Generation.

\end{document}
"""

    splitter = LatexTextSplitter(chunk_size=200, chunk_overlap=0)
    chunks = splitter.split_text(latex_text)

    print(f"chunk_size=200, 切分为 {len(chunks)} 个块:")
    for i, chunk in enumerate(chunks):
        preview = chunk.replace("\n", " ").strip()
        print(f"\n  块 {i+1}（{len(chunk)} 字符）:")
        print(f"    {preview}")

    print("\n特点:")
    print("  - 按 LaTeX 结构（\\section, \\begin{...} 等）切分")
    print("  - 保持 LaTeX 环境的完整性（不会在 \\begin/\\end 中间断开）")
    print("  - 适合学术论文、技术报告等 LaTeX 文档")


if __name__ == "__main__":
    demo_latex_splitter()
