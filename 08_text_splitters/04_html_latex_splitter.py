"""
04_html_latex_splitter.py - HTMLSectionSplitter + LatexTextSplitter

HTMLSectionSplitter: 按 HTML 标签（h1-h6）切分，保留标题作为 metadata。
LatexTextSplitter: 按 LaTeX 语义边界（章节、环境）切分。

参考文档：
  - HTMLSectionSplitter: https://reference.langchain.com/python/langchain-text-splitters/text_splitters/html/HTMLSectionSplitter
  - LatexTextSplitter: https://reference.langchain.com/python/langchain-text-splitters/text_splitters/latex/LatexTextSplitter

安装：
  pip install langchain-text-splitters
"""

from langchain_text_splitters import LatexTextSplitter

try:
    from langchain_text_splitters import HTMLSectionSplitter
    _HAS_LXML = True
except ImportError:
    _HAS_LXML = False


# ============================================================
# 演示 1：HTMLSectionSplitter（按 HTML 标题标签切分）
# ============================================================

def demo_html_section():
    """
    HTMLSectionSplitter 按 HTML 标签（h1-h6）切分，
    将标题文本存入 metadata。

    类似 MarkdownHeaderTextSplitter，但针对 HTML。
    适合：网页内容、HTML 文档、爬虫结果等。
    """
    print("=== 演示 1：HTMLSectionSplitter ===")
    if not _HAS_LXML:
        print("缺少依赖: lxml 和 bs4（HTMLSectionSplitter 需要）")
        print("安装命令: pip install lxml bs4")
        return

    html_text = """
    <html>
    <body>
        <h1>LangChain 教程</h1>
        <p>LangChain 是一个用于构建 LLM 应用的框架。</p>

        <h2>第一章：基础概念</h2>
        <p>Models、Prompts 和 Chains 是三大核心组件。</p>

        <h3>1.1 Models</h3>
        <p>Chat Models 是 LangChain 1.x 的唯一推荐模型接口。</p>

        <h3>1.2 Prompts</h3>
        <p>PromptTemplate 和 ChatPromptTemplate 用于管理提示词。</p>

        <h2>第二章：RAG 系统</h2>
        <p>RAG = 文档加载 + 文本切分 + 向量存储 + 检索 + 生成。</p>
    </body>
    </html>
    """

    splitter = HTMLSectionSplitter(
        headers_to_split_on=[
            ("h1", "h1"),
            ("h2", "h2"),
            ("h3", "h3"),
        ],
    )
    try:
        docs = splitter.split_text(html_text)
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("安装命令: pip install lxml bs4")
        return

    print(f"切分为 {len(docs)} 个块:")
    for i, doc in enumerate(docs):
        content_preview = doc.page_content[:80].replace("\n", " ").strip()
        print(f"\n  块 {i+1}:")
        print(f"    metadata: {doc.metadata}")
        print(f"    content:  {content_preview}...")

    print("\n特点:")
    print("  - 按 HTML 标题标签（h1-h6）切分，保留标题层级为 metadata")
    print("  - 自动去除 HTML 标签，只保留文本内容")
    print("  - 适合网页内容的结构化提取")


# ============================================================
# 演示 2：LatexTextSplitter（按 LaTeX 语义切分）
# ============================================================

def demo_latex():
    """
    LatexTextSplitter 按 LaTeX 文档结构切分：
    - 章节（\\section, \\subsection, \\subsubsection）
    - 环境（\\begin{...} ... \\end{...}）
    - 列表、表格等语义边界

    适合：学术论文、技术报告等 LaTeX 格式文档。
    """
    print("\n\n=== 演示 2：LatexTextSplitter ===")

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
        preview = chunk[:80].replace("\n", " ").strip()
        print(f"\n  块 {i+1}（{len(chunk)} 字符）:")
        print(f"    {preview}...")

    print("\n特点:")
    print("  - 按 LaTeX 结构（\\section, \\begin{...} 等）切分")
    print("  - 保持 LaTeX 环境的完整性（不会在 \\begin/\\end 中间断开）")
    print("  - 适合学术论文、技术报告等 LaTeX 文档")


if __name__ == "__main__":
    demo_html_section()
    demo_latex()
