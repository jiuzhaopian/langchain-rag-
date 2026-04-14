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
# 演示 4：PDF 图片提取 + OCR（PyPDFLoader extract_images）
# ============================================================

def _patch_pypdf_extract_images():
    """
    修复 langchain-community PyPDFParser.extract_images_from_page 的 bug：
    在写入图片数据之前检查 BytesIO 是否为空（永远为 True），导致所有图片被跳过。
    参考: https://github.com/langchain-ai/langchain/issues/34400

    此函数替换有 bug 的方法，去掉无效的空缓冲区检查。
    """
    import io
    import numpy as np
    import pypdf
    from PIL import Image

    from langchain_community.document_loaders.parsers.pdf import (
        PyPDFParser,
        _FORMAT_IMAGE_STR,
        _JOIN_IMAGES,
        _format_inner_image,
        _PDF_FILTER_WITHOUT_LOSS,
        _PDF_FILTER_WITH_LOSS,
    )
    from langchain_core.documents.base import Blob

    def patched_extract(self, page):
        """修复版：去掉空缓冲区检查，正常写入数据后再创建 Blob。"""
        if not self.images_parser:
            return ""
        if "/XObject" not in page["/Resources"].keys():
            return ""
        xObject = page["/Resources"]["/XObject"].get_object()
        images = []
        for obj in xObject:
            if xObject[obj]["/Subtype"] == "/Image":
                img_filter = (
                    xObject[obj]["/Filter"][1:]
                    if type(xObject[obj]["/Filter"]) is pypdf.generic._base.NameObject
                    else xObject[obj]["/Filter"][0][1:]
                )
                np_image = None
                if img_filter in _PDF_FILTER_WITHOUT_LOSS:
                    height = int(xObject[obj]["/Height"])
                    width = int(xObject[obj]["/Width"])
                    np_image = np.frombuffer(
                        xObject[obj].get_data(), dtype=np.uint8
                    ).reshape(height, width, -1)
                elif img_filter in _PDF_FILTER_WITH_LOSS:
                    np_image = np.array(
                        Image.open(io.BytesIO(xObject[obj].get_data()))
                    )
                if np_image is not None:
                    image_bytes = io.BytesIO()
                    Image.fromarray(np_image).save(image_bytes, format="PNG")
                    blob = Blob.from_data(
                        image_bytes.getvalue(), mime_type="image/png"
                    )
                    image_text = next(
                        self.images_parser.lazy_parse(blob)
                    ).page_content
                    images.append(
                        _format_inner_image(
                            blob, image_text, self.images_inner_format
                        )
                    )
        return _FORMAT_IMAGE_STR.format(
            image_text=_JOIN_IMAGES.join(filter(None, images))
        )

    PyPDFParser.extract_images_from_page = patched_extract


def demo_extract_images(pdf_path):
    """
    PyPDFLoader 的 extract_images=True 会提取 PDF 中嵌入的图片，
    并通过 images_parser 对图片做 OCR 识别。

    默认使用 RapidOCRBlobParser（无需额外配置），
    也可用 LLMImageBlobParser（调用多模态大模型识别）。

    images_inner_format 控制图片区域的输出格式：
      - "text"（默认）: 仅输出 OCR 识别的文字
      - "markdown-img": Markdown 图片标签 + OCR 文字
      - "html-img": HTML <img> 标签 + OCR 文字

    ⚠️ 已知 bug: langchain-community 的 PyPDFParser 在写入图片数据之前
       错误地检查 BytesIO 是否为空（永远为 True），导致所有图片被跳过。
       本 demo 通过 monkey-patch 修复此问题。
       参考: https://github.com/langchain-ai/langchain/issues/34400
       等官方修复后可移除 _patch_pypdf_extract_images() 调用。

    依赖: pip install pypdf rapidocr-onnxruntime
    """
    print("\n=== 演示 4：PDF 图片提取 + OCR ===")

    try:
        from langchain_community.document_loaders.parsers import RapidOCRBlobParser
    except ImportError:
        print("跳过: 需要 pip install rapidocr-onnxruntime")
        return

    # 修复已知 bug
    _patch_pypdf_extract_images()

    # --- 4a: 默认 RapidOCRBlobParser ---
    print("\n--- 4a: PyPDFLoader(extract_images=True) + RapidOCRBlobParser ---")
    loader = PyPDFLoader(pdf_path, extract_images=True)
    docs = loader.load()
    for d in docs:
        print(f"Page {d.metadata.get('page')}:")
        print(d.page_content)

    # --- 4b: images_inner_format 对比 ---
    print("\n--- 4b: images_inner_format 三种格式 ---")
    for fmt in ["text", "markdown-img", "html-img"]:
        loader = PyPDFLoader(
            pdf_path,
            extract_images=True,
            images_inner_format=fmt,
        )
        docs = loader.load()
        content = docs[0].page_content
        if fmt == "text":
            print(f"  [{fmt}]:")
            for line in content.strip().split("\n"):
                print(f"    {line}")
        else:
            # 只展示格式标签部分
            tag_line = [l for l in content.split("\n") if "!" in l or "<img" in l]
            print(f"  [{fmt}]: {tag_line[0][:80]}... + OCR文字")

    # --- 4c: LLMImageBlobParser（示例，需 API Key） ---
    print("\n--- 4c: LLMImageBlobParser（需 API Key，仅展示用法）---")
    print("  from langchain_community.document_loaders.parsers import LLMImageBlobParser")
    print("  from langchain_openai import ChatOpenAI")
    print("  loader = PyPDFLoader(")
    print("      'invoice.pdf',")
    print("      extract_images=True,")
    print("      images_parser=LLMImageBlobParser(")
    print("          model=ChatOpenAI(model='gpt-4o-mini', max_tokens=1024)")
    print("      ),")
    print("  )")


if __name__ == "__main__":
    pdf_path = os.path.join(DATA_DIR, "报销制度.pdf")
    if os.path.exists(pdf_path):
        demo_basic(pdf_path)
        demo_lazy_load(pdf_path)
    demo_layout_mode()
    demo_extract_images(os.path.join(DATA_DIR, "test_invoice.pdf"))