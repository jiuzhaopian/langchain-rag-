"""
Test: PyMuPDF image extraction + OCR pipeline
"""
import fitz  # pymupdf
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PIL import Image, ImageDraw, ImageFont
import os

WORKSPACE = os.path.dirname(os.path.abspath(__file__))

def create_test_pdf():
    """Create a PDF with an embedded invoice image."""
    # Create image with text
    img = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    draw.text((30, 30), "Invoice #2024-001", fill="black", font=font)
    draw.text((30, 80), "Total: 12800.00 RMB", fill="black", font=font)
    draw.text((30, 130), "Date: 2024-03-15", fill="black", font=font)
    img_path = os.path.join(WORKSPACE, "test_invoice.png")
    img.save(img_path)
    print(f"[1] Image created: {img_path}")

    # Create PDF
    pdf_path = os.path.join(WORKSPACE, "test_invoice.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.drawImage(img_path, 100, 300, width=400, height=200)
    c.drawString(100, 550, "This PDF contains an embedded invoice image:")
    c.save()
    print(f"[2] PDF created: {pdf_path}")
    return pdf_path, img_path


def test_pymupdf_extraction(pdf_path):
    """Test PyMuPDF image extraction."""
    print("\n" + "=" * 60)
    print("=== Test 1: PyMuPDF extract_image ===")
    print("=" * 60)
    doc = fitz.open(pdf_path)
    page = doc[0]
    images = page.get_images(full=True)
    print(f"Found {len(images)} image(s)")

    for idx, img_info in enumerate(images):
        xref = img_info[0]
        bi = doc.extract_image(xref)
        img_ext = bi["ext"]
        img_bytes = bi["image"]
        extracted_path = os.path.join(WORKSPACE, f"extracted_{idx}.{img_ext}")
        with open(extracted_path, "wb") as f:
            f.write(img_bytes)
        print(f"  [{idx}] ext={img_ext}, size={len(img_bytes)} bytes, "
              f"{bi['width']}x{bi['height']}, saved={extracted_path}")
    doc.close()
    return os.path.join(WORKSPACE, f"extracted_0.{img_ext}")


def test_ocr(image_path, label):
    """Test OCR on a given image."""
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    result, _ = ocr(image_path)
    print(f"\n--- OCR: {label} ---")
    if result:
        for line in result:
            text = line[1]
            conf = line[2]
            print(f"  \"{text}\" (conf: {conf:.3f})")
        return True
    else:
        print("  No text detected!")
        return False


def test_pymupdf_loader(pdf_path):
    """Test LangChain PyMuPDFLoader."""
    from langchain_community.document_loaders import PyMuPDFLoader
    print("\n" + "=" * 60)
    print("=== Test 3: PyMuPDFLoader ===")
    print("=" * 60)
    docs = PyMuPDFLoader(pdf_path).load()
    for i, d in enumerate(docs):
        print(f"Page {i}: {repr(d.page_content[:300])}")


if __name__ == "__main__":
    pdf_path, img_path = create_test_pdf()
    extracted_path = test_pymupdf_extraction(pdf_path)
    
    print("\n" + "=" * 60)
    print("=== Test 2: OCR comparison ===")
    print("=" * 60)
    ok_original = test_ocr(img_path, "Original PNG")
    ok_extracted = test_ocr(extracted_path, "PyMuPDF extracted image")
    
    print("\n" + "=" * 60)
    print("=== Summary ===")
    print("=" * 60)
    print(f"OCR on original PNG:      {'PASS' if ok_original else 'FAIL'}")
    print(f"OCR on extracted image:   {'PASS' if ok_extracted else 'FAIL'}")
    
    test_pymupdf_loader(pdf_path)
