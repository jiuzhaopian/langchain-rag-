"""
05_pdf_loader.py - PDF 加载器

PyPDFLoader 将 PDF 文件每页加载为一个 Document。
需要安装 pypdf 包（LangChain 社区版 PDF 加载的默认依赖）。

参考文档：
  - PyPDFLoader: https://docs.langchain.com/oss/python/integrations/document_loaders/pypdfloader

安装：
  pip install langchain-community pypdf
"""

import os

from langchain_community.document_loaders import PyPDFLoader

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def create_sample_pdf():
    """创建示例 PDF 文件（如果没有）"""
    os.makedirs(DATA_DIR, exist_ok=True)
    pdf_path = os.path.join(DATA_DIR, "sample.pdf")

    if os.path.exists(pdf_path):
        return pdf_path

    # 用 fpdf2 生成示例 PDF
    try:
        from fpdf import FPDF
    except ImportError:
        print("需要 fpdf2 来生成示例 PDF: pip install fpdf2")
        return None

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


# ============================================================
# 演示 1：基本用法 - 每页一个 Document
# ============================================================

def demo_basic(pdf_path):
    """
    PyPDFLoader 默认每页生成一个 Document。
    """
    print("=== 演示 1：PyPDFLoader 基本用法 ===")

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

    loader = PyPDFLoader(pdf_path)

    print("逐页懒加载:")
    for i, doc in enumerate(loader.lazy_load()):
        print(f"  第 {i+1} 页: {len(doc.page_content)} 字符")


# ============================================================
# 演示 3：layout 提取模式
# ============================================================

def demo_layout_mode():
    """
    extraction_mode="layout" 按页面渲染布局提取文本，
    保留表格的固定宽度对齐格式，更贴近 PDF 的视觉效果。
    extraction_mode="plain"（默认）按文本流提取，表格格式会丢失。
    """
    print("\n=== 演示 3：plain vs layout 提取模式 ===")

    pdf_path = os.path.join(DATA_DIR, "报销制度.pdf")
    if not os.path.exists(pdf_path):
        print(f"跳过: {pdf_path} 不存在")
        return

    # plain 模式（默认）
    loader_plain = PyPDFLoader(pdf_path, extraction_mode="plain")
    doc_plain = loader_plain.load()[0]

    # layout 模式
    loader_layout = PyPDFLoader(pdf_path, extraction_mode="layout")
    doc_layout = loader_layout.load()[0]

    print(f"plain 模式第1页: {len(doc_plain.page_content)} 字符")
    print(doc_plain.page_content[:300])
    print(f"\nlayout 模式第1页: {len(doc_layout.page_content)} 字符")
    print(doc_layout.page_content[:300])
    print("\n对比: layout 模式保留了表格的列对齐格式，plain 模式丢失了表格结构")


# ============================================================
# 演示 4：extract_images 图片提取
# ============================================================

def demo_extract_images():
    """
    extract_images=True 提取 PDF 中的嵌入图片。
    images_inner_format 控制图片在文本中的输出格式：
      - "text"（默认）: 图片内容原样输出
      - "markdown-img": 包装为 ![alt](#) 格式
      - "html-img": 包装为 <img alt="body" src="#"/> 格式
    images_parser 可选配 OCR 解析器（如 RapidOCRBlobParser）识别图片中的文字。
    """
    print("\n=== 演示 4：extract_images 图片提取 ===")
    print("extract_images=True 需要额外依赖（如 rapidocr-onnxruntime），沙箱未安装")
    print("示例代码（需要 pip install rapidocr-onnxruntime）:")
    print()
    print('  from langchain_community.document_loaders import PyPDFLoader')
    print('  from langchain_community.document_loaders.blob_loaders import BlobLoader')
    print('  from langchain_community.document_loaders.parsers.images import RapidOCRBlobParser')
    print()
    print('  loader = PyPDFLoader(')
    print('      "报销制度.pdf",')
    print('      extract_images=True,')
    print('      images_parser=RapidOCRBlobParser(),')
    print('      images_inner_format="markdown-img",  # 图片输出为 ![alt](#)')
    print('  )')
    print('  docs = loader.load()')


if __name__ == "__main__":
    pdf_path = create_sample_pdf()
    if pdf_path and os.path.exists(pdf_path):
        demo_basic(pdf_path)
        demo_lazy_load(pdf_path)
    demo_layout_mode()
    demo_extract_images()