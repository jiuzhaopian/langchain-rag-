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
# 演示 4：PDF 图片提取 + OCR
# ============================================================

def demo_extract_images():
    """
    从 PDF 中提取嵌入图片并通过 OCR 识别图片文字。

    方案：PyMuPDF (fitz) 提取图片 + RapidOCR 识别 + 手动拼装 Document。
    原因：PyPDFLoader 的 extract_images 在 pypdf 6.10.0 下存在图片数据丢失问题，
    提取后的图片经 OCR 输出为空（验证详见 extract_images_research/README.md）。

    images_inner_format 控制图片输出格式：
      - "text"（默认）: 纯 OCR 文字
      - "markdown-img": ![...] base64 图片 + OCR 文字
      - "html-img": <img src=base64> + OCR 文字

    依赖: pip install pymupdf rapidocr-onnxruntime reportlab Pillow
    """
    print("\n=== 演示 4：PDF 图片提取 + OCR ===")

    try:
        import fitz  # pymupdf
    except ImportError:
        print("跳过: 需要 pip install pymupdf")
        return

    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        print("跳过: 需要 pip install rapidocr-onnxruntime")
        return

    # --- 4a: 提取 PDF 中的图片 ---
    print("\n--- 4a: PyMuPDF 提取 PDF 图片 ---")
    img_path = os.path.join(DATA_DIR, "test_invoice.png")
    pdf_path = os.path.join(DATA_DIR, "test_invoice.pdf")

    if not os.path.exists(pdf_path):
        # 创建含图片的测试 PDF
        try:
            from PIL import Image, ImageDraw
            import reportlab.lib.pagesizes as ps
            from reportlab.pdfgen import canvas
        except ImportError:
            print("跳过: 需要 reportlab + Pillow")
            return

        os.makedirs(DATA_DIR, exist_ok=True)
        # 生成发票图片
        img = Image.new("RGB", (400, 200), color="white")
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 399, 199], outline="black", width=2)
        draw.text((30, 30), "Invoice #2024-001", fill="black")
        draw.text((30, 80), "Total: 12800.00 RMB", fill="black")
        draw.text((30, 130), "Date: 2024-03-15", fill="black")
        img.save(img_path)

        c = canvas.Canvas(pdf_path, pagesize=ps.A4)
        c.setFont("Helvetica", 14)
        c.drawString(100, 700, "This PDF contains an embedded invoice image:")
        c.drawImage(img_path, 100, 400, width=400, height=200)
        c.showPage()
        c.save()
        print(f"已生成测试 PDF: {pdf_path}")

    # 提取图片
    pdf_doc = fitz.open(pdf_path)
    page = pdf_doc[0]
    images = page.get_images(full=True)
    print(f"共找到 {len(images)} 个嵌入图片")

    for idx, img_info in enumerate(images):
        xref = img_info[0]
        bi = pdf_doc.extract_image(xref)
        print(f"  [{idx}] 格式={bi['ext']}, 大小={len(bi['image'])}字节")

    pdf_doc.close()

    # --- 4b: OCR 识别 ---
    print("\n--- 4b: OCR 识别提取的图片 ---")
    ocr = RapidOCR()

    # 对原图 OCR
    with open(img_path, "rb") as f:
        original_bytes = f.read()

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(original_bytes)
        tmp_path = f.name
    result_orig, _ = ocr(tmp_path)
    os.unlink(tmp_path)
    print(f"原图 OCR: {[line[1] for line in result_orig] if result_orig else []}")

    # 对 PyMuPDF 提取的图片 OCR
    pdf_doc = fitz.open(pdf_path)
    xref = images[0][0]
    bi = pdf_doc.extract_image(xref)
    extracted_bytes = bi["image"]
    pdf_doc.close()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(extracted_bytes)
        tmp_path = f.name
    result_ext, _ = ocr(tmp_path)
    os.unlink(tmp_path)
    print(f"提取图 OCR: {[line[1] for line in result_ext] if result_ext else []}")
    print(f"结果一致: {result_orig == result_ext}")

    # --- 4c: images_inner_format 三种格式 ---
    print("\n--- 4c: images_inner_format 三种格式 ---")
    import base64

    ocr_text = "\n".join(line[1] for line in result_ext) if result_ext else ""
    b64 = base64.b64encode(extracted_bytes).decode()

    formats = {
        "text": ocr_text,
        "markdown-img": f"![Extracted image](data:image/png;base64,{b64})\n{ocr_text}",
        "html-img": f'<img src="data:image/png;base64,{b64}" />\n{ocr_text}',
    }
    for fmt, content in formats.items():
        preview = content[:100].replace("\n", " ")
        if fmt != "text":
            preview = content.split("\n")[0][:60] + f"... + OCR文字"
        print(f"  [{fmt}] {preview}")


if __name__ == "__main__":
    pdf_path = create_sample_pdf()
    if pdf_path and os.path.exists(pdf_path):
        demo_basic(pdf_path)
        demo_lazy_load(pdf_path)
    demo_layout_mode()
    demo_extract_images()