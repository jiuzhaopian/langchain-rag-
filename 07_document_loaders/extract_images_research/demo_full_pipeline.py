"""
方案：PyMuPDF 提取图片 + RapidOCR + 手动拼装 LangChain Document
展示 images_inner_format 三种格式（text / markdown-img / html-img）

结论：PyPDFLoader 的 extract_images 在当前版本有 bug（pypdf reshape 丢失数据），
     用 PyMuPDF 替代可以完美跑通。
"""
import os
import base64
import fitz  # pymupdf
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PIL import Image, ImageDraw, ImageFont
from rapidocr_onnxruntime import RapidOCR
from langchain_core.documents import Document

WORKSPACE = os.path.dirname(os.path.abspath(__file__))


def create_test_pdf():
    """Create a PDF with an embedded invoice image."""
    img = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    draw.text((30, 30), "Invoice #2024-001", fill="black", font=font)
    draw.text((30, 80), "Total: 12800.00 RMB", fill="black", font=font)
    draw.text((30, 130), "Date: 2024-03-15", fill="black", font=font)
    img_path = os.path.join(WORKSPACE, "test_invoice.png")
    img.save(img_path)

    pdf_path = os.path.join(WORKSPACE, "test_invoice.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.drawImage(img_path, 100, 300, width=400, height=200)
    c.drawString(100, 550, "This PDF contains an embedded invoice image:")
    c.save()
    return pdf_path


def ocr_image(image_bytes):
    """Run OCR on image bytes, return text."""
    ocr = RapidOCR()
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(image_bytes)
        tmp_path = f.name
    result, _ = ocr(tmp_path)
    os.unlink(tmp_path)
    if result:
        return "\n".join(line[1] for line in result)
    return ""


def format_image_content(ocr_text, image_bytes, fmt):
    """Format image content in different styles (mimicking images_inner_format)."""
    b64 = base64.b64encode(image_bytes).decode()
    if fmt == "text":
        # Pure text: OCR result only
        return ocr_text if ocr_text.strip() else "[No OCR text detected]"
    elif fmt == "markdown-img":
        # Markdown image tag + OCR text
        return f"![Extracted image](data:image/png;base64,{b64})\n{ocr_text}"
    elif fmt == "html-img":
        # HTML img tag + OCR text
        return f'<img src="data:image/png;base64,{b64}" />\n{ocr_text}'
    return ocr_text


def load_pdf_with_ocr(pdf_path, images_inner_format="text"):
    """Load PDF with PyMuPDF, extract images, OCR them, assemble Documents."""
    doc = fitz.open(pdf_path)
    documents = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Get text content
        text = page.get_text().strip()

        # Get images and OCR them
        image_parts = []
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            bi = doc.extract_image(xref)
            img_bytes = bi["image"]
            ocr_text = ocr_image(img_bytes)
            formatted = format_image_content(ocr_text, img_bytes, images_inner_format)
            image_parts.append(formatted)

        # Combine: text + image OCR results
        all_parts = []
        if text:
            all_parts.append(text)
        all_parts.extend(image_parts)
        page_content = "\n\n".join(all_parts)

        documents.append(Document(
            page_content=page_content,
            metadata={"source": pdf_path, "page": page_num + 1}
        ))
    doc.close()
    return documents


if __name__ == "__main__":
    pdf_path = create_test_pdf()

    print("=" * 60)
    print("=== images_inner_format='text' ===")
    print("=" * 60)
    docs = load_pdf_with_ocr(pdf_path, "text")
    for d in docs:
        print(d.page_content)

    print("\n" + "=" * 60)
    print("=== images_inner_format='markdown-img' ===")
    print("=" * 60)
    docs = load_pdf_with_ocr(pdf_path, "markdown-img")
    for d in docs:
        # Truncate base64 for display
        content = d.page_content
        # Show first 100 chars of base64 data
        import re
        content_short = re.sub(
            r'(data:image/png;base64,)[A-Za-z0-9+/=]{50,}',
            r'\1[...base64 truncated...]',
            content
        )
        print(content_short)

    print("\n" + "=" * 60)
    print("=== images_inner_format='html-img' ===")
    print("=" * 60)
    docs = load_pdf_with_ocr(pdf_path, "html-img")
    for d in docs:
        content = d.page_content
        import re
        content_short = re.sub(
            r'(data:image/png;base64,)[A-Za-z0-9+/=]{50,}',
            r'\1[...base64 truncated...]',
            content
        )
        print(content_short)

    print("\n" + "=" * 60)
    print("=== 总结 ===")
    print("=" * 60)
    print("PyMuPDF 提取图片 -> OCR -> 拼装 Document: 全链路跑通")
    print("三种格式差异:")
    print("  text:         纯 OCR 文字")
    print("  markdown-img: ![...] base64 图片 + OCR 文字")
    print("  html-img:     <img src=base64> + OCR 文字")
