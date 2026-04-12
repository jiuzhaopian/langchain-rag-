"""
04_pdf_loader.py - PDF 加载器

PyPDFLoader 将 PDF 文件每页加载为一个 Document。
需要安装 pypdf 包（LangChain 社区版 PDF 加载的默认依赖）。

参考文档：
  - PyPDFLoader: https://python.langchain.com/docs/integrations/document_loaders/pypdf/

安装：
  pip install langchain-community pypdf
"""

import os


def create_sample_pdf():
    """创建示例 PDF 文件（如果没有）"""
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = "sample.pdf"

    if os.path.exists(pdf_path):
        return pdf_path

    # 用 fpdf2 生成示例 PDF
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=16)
        pdf.cell(0, 10, "LangChain Study Guide", ln=True, align="C")
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8, "Chapter 1: Introduction\nLangChain is a framework for building LLM applications.")
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8, "Chapter 2: Core Concepts\nModels, Prompts, Output Parsers, Chains, Retrievers.")
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8, "Chapter 3: RAG\nRetrieval Augmented Generation - the core pattern for AI knowledge.")
        pdf.output(pdf_path)
        return pdf_path
    except ImportError:
        print("需要 fpdf2 来生成示例 PDF: pip install fpdf2")
        return None


# ============================================================
# 演示 1：基本用法 - 每页一个 Document
# ============================================================

def demo_basic(pdf_path):
    """
    PyPDFLoader 默认每页生成一个 Document。
    """
    print("=== 演示 1：PyPDFLoader 基本用法 ===")

    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    print(f"PDF 共 {len(docs)} 页")
    for i, doc in enumerate(docs):
        content_preview = doc.page_content[:80].replace("\n", " ")
        print(f"  第 {i+1} 页: {content_preview}...")
        print(f"    元数据: page={doc.metadata.get('page')}, source={os.path.basename(doc.metadata.get('source', ''))}")


# ============================================================
# 演示 2：懒加载
# ============================================================

def demo_lazy_load(pdf_path):
    """
    懒加载适合大 PDF 文件，逐页处理。
    """
    print("\n=== 演示 2：懒加载 ===")

    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(pdf_path)

    print("逐页懒加载:")
    for i, doc in enumerate(loader.lazy_load()):
        print(f"  第 {i+1} 页: {len(doc.page_content)} 字符")


if __name__ == "__main__":
    pdf_path = create_sample_pdf()
    if pdf_path and os.path.exists(pdf_path):
        demo_basic(pdf_path)
        demo_lazy_load(pdf_path)
