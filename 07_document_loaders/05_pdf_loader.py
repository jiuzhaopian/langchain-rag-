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
    extract_images=True 提取 PDF 中的嵌入图片对象。
    必须配合 images_parser 使用，不设 parser 时 extract_images 无效（直接返回空）。

    images_inner_format 控制图片输出格式：
      - "text"（默认）: OCR 结果原样嵌入文本
      - "markdown-img": 包装为 ![OCR文字](#)
      - "html-img": 包装为 <img alt="OCR文字" src="#"/>

    images_parser: 图片解析器，常用：
      - RapidOCRBlobParser() — 基于 PaddleOCR，需 pip install rapidocr-onnxruntime
      - TesseractBlobParser() — 需系统安装 tesseract

    依赖: pip install pypdf rapidocr-onnxruntime
    """
    print("\n=== 演示 4：extract_images 图片提取 ===")

    try:
        from langchain_community.document_loaders.parsers.images import RapidOCRBlobParser
        from langchain_community.document_loaders.blob_loaders import Blob
    except ImportError:
        print("跳过: 需要 pip install rapidocr-onnxruntime")
        return

    img_path = os.path.join(DATA_DIR, "test_invoice.png")
    if not os.path.exists(img_path):
        print(f"跳过: {img_path} 不存在")
        return

    # --- 4a: RapidOCRBlobParser 直接 OCR 图片文件 ---
    print("--- 4a: RapidOCRBlobParser 直接 OCR 图片 ---")
    with open(img_path, "rb") as f:
        blob = Blob.from_data(f.read(), mime_type="image/png")
    ocr_doc = list(RapidOCRBlobParser().lazy_parse(blob))[0]
    print(f"OCR 识别结果: {repr(ocr_doc.page_content)}")

    # --- 4b: PyPDFLoader extract_images 参数说明 ---
    # 实测发现: pypdf 从 PDF XObject 提取图片后 reshape 会导致图片数据质量下降，
    # RapidOCRBlobParser OCR 后输出为空。这是 pypdf 当前版本的已知问题。
    # 直接对原始图片文件做 OCR（如 4a）可以正常工作。
    pdf_path = os.path.join(DATA_DIR, "test_with_image.pdf")
    if os.path.exists(pdf_path):
        print("\n--- 4b: PyPDFLoader extract_images 实测 ---")
        loader = PyPDFLoader(
            pdf_path,
            extract_images=True,
            images_parser=RapidOCRBlobParser(),
        )
        doc = loader.load()[0]
        print(f"extract_images=True + OCR: {len(doc.page_content)} 字符")
        print(f"  内容: {repr(doc.page_content)}")
        print("  注: pypdf 提取图片后 OCR 结果为空（图片质量在提取过程中丢失），")
        print("      对比 4a 直接 OCR 原图可以正常识别。")

    # --- 4c: images_inner_format 三种格式对比 ---
    # 以下展示参数用法（OCR 有结果时三种格式输出不同）
    print("\n--- 4c: images_inner_format 参数说明 ---")
    print('  "text" (默认):     OCR 文字原样嵌入: Invoice #2024-001')
    print('  "markdown-img":    ![Invoice #2024-001](#)')
    print('  "html-img":        <img alt="Invoice #2024-001" src="#"/>')
    print("  依赖 OCR 返回非空内容才能看到差异。")


def create_test_pdf():
    """生成含嵌入图片的测试 PDF（用于 demo_extract_images）"""
    os.makedirs(DATA_DIR, exist_ok=True)
    pdf_path = os.path.join(DATA_DIR, "test_with_image.pdf")
    img_path = os.path.join(DATA_DIR, "test_invoice.png")

    if os.path.exists(pdf_path) and os.path.exists(img_path):
        return

    try:
        from PIL import Image, ImageDraw
        import reportlab.lib.pagesizes as ps
        from reportlab.pdfgen import canvas
    except ImportError:
        print("需要 reportlab + Pillow: pip install reportlab Pillow")
        return

    # 生成发票图片
    img = Image.new("RGB", (400, 120), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 399, 119], outline="black", width=2)
    draw.text((20, 15), "Invoice #2024-001", fill="black")
    draw.text((20, 45), "Total: 12800.00 RMB", fill="black")
    draw.text((20, 75), "Date: 2024-03-15", fill="black")
    img.save(img_path)

    # 用 reportlab 生成 PDF（图片作为 XObject 嵌入，pypdf 可提取）
    c = canvas.Canvas(pdf_path, pagesize=ps.A4)
    c.setFont("Helvetica", 14)
    c.drawString(100, 750, "Document with Embedded Images")
    c.setFont("Helvetica", 11)
    c.drawString(100, 720, "Below is a simulated invoice image.")
    c.drawImage(img_path, 100, 550, width=400, height=120)
    c.showPage()
    c.setFont("Helvetica", 11)
    c.drawString(100, 750, "Page 2: text only, no images.")
    c.showPage()
    c.save()


if __name__ == "__main__":
    pdf_path = create_sample_pdf()
    if pdf_path and os.path.exists(pdf_path):
        demo_basic(pdf_path)
        demo_lazy_load(pdf_path)
    demo_layout_mode()
    create_test_pdf()
    demo_extract_images()