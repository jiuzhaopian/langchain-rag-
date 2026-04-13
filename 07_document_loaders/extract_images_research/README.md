# PyPDFLoader extract_images 研究与替代方案

## 问题

PyPDFLoader 的 `extract_images=True` + `RapidOCRBlobParser()` 在 pypdf 当前版本（6.10.0）存在问题：

1. **pypdf 从 PDF XObject 提取图片后 reshape 导致数据质量丢失**，OCR 识别结果为空
2. **直接对原始 PNG 做 OCR 正常**（`Invoice #2024-001 / Total 12800.00 RMB / Date: 2024-03-15`）
3. 问题出在 pypdf 提取环节，非 OCR 问题

## 测试环境

- Docker: `code-sandbox:latest`
- pypdf: 6.10.0
- pymupdf (fitz): 最新版
- rapidocr-onnxruntime: 最新版

## 测试结果

### PyPDFLoader（❌ 不可用）

```
pypdf version: 6.10.0
Pages: 1
Content: 'This PDF contains an embedded invoice image:'
# 图片内容完全丢失，OCR 无输出
```

### PyMuPDF 提取 + OCR（✅ 完美通过）

```
=== PyMuPDF extract_image ===
Found 1 image(s)
  [0] ext=png, size=8842 bytes, 400x200

--- OCR: Original PNG ---
  "Invoice#2024-001" (conf: 0.971)
  "Total:12800.00" (conf: 0.986)
  "RMB" (conf: 0.996)
  "Date:2024-03-15" (conf: 0.993)

--- OCR: PyMuPDF extracted image ---
  "Invoice#2024-001" (conf: 0.971)
  "Total:12800.00" (conf: 0.986)
  "RMB" (conf: 0.996)
  "Date:2024-03-15" (conf: 0.993)
```

PyMuPDF 提取的图片与原图 OCR 结果**完全一致**，无数据丢失。

## 替代方案

### 方案：PyMuPDF + RapidOCR + 手动拼装 Document

用 PyMuPDF 替代 PyPDFLoader 做图片提取，再用 RapidOCR 做 OCR，最后手动拼装 LangChain Document 对象。

核心代码见 `demo_full_pipeline.py`，支持三种 `images_inner_format` 风格：

| 格式 | 输出 |
|------|------|
| `text` | 纯 OCR 文字 |
| `markdown-img` | `![...]` base64 图片 + OCR 文字 |
| `html-img` | `<img src=base64>` + OCR 文字 |

### 关键步骤

```python
import fitz  # pymupdf
from rapidocr_onnxruntime import RapidOCR
from langchain_core.documents import Document

doc = fitz.open("invoice.pdf")
for page in doc:
    # 1. 提取文字
    text = page.get_text()
    # 2. 提取图片
    for img_info in page.get_images(full=True):
        bi = doc.extract_image(img_info[0])
        # 3. OCR
        result, _ = ocr(bi["image"])
    # 4. 拼装 Document
    documents.append(Document(page_content=..., metadata=...))
```

## 依赖

```
pymupdf
rapidocr-onnxruntime
langchain-core
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `test_pymupdf.py` | 验证 PyMuPDF 图片提取 + OCR 链路 |
| `demo_full_pipeline.py` | 完整方案：三种 images_inner_format 演示 |
| `README.md` | 本文件 |
