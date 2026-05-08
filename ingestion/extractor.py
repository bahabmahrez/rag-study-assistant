import fitz  # PyMuPDF
from pathlib import Path


def extract_with_pymupdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        pages.append({
            "page": page_num,
            "text": text,
            "source": Path(pdf_path).name,
        })

    doc.close()
    return pages


def check_math_quality(text: str) -> float:
    if not text.strip():
        return 1.0
    broken_indicators = ["□", "■", "\ufffd"]
    broken_count = sum(text.count(ind) for ind in broken_indicators)
    return broken_count / max(len(text), 1)


def extract_pdf(pdf_path: str) -> list[dict]:
    print(f"Extracting: {pdf_path}")
    pages = extract_with_pymupdf(pdf_path)

    flagged = 0
    for page in pages:
        score = check_math_quality(page["text"])
        page["quality_score"] = score
        page["needs_fallback"] = score > 0.01
        if page["needs_fallback"]:
            flagged += 1

    print(f"  → {len(pages)} pages, {flagged} flagged")
    return pages


def extract_all_pdfs(pdf_dir: str) -> list[dict]:
    all_pages = []
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {pdf_dir}")
        return []

    for pdf_path in pdf_files:
        pages = extract_pdf(str(pdf_path))
        all_pages.extend(pages)

    print(f"\nTotal: {len(all_pages)} pages from {len(pdf_files)} PDFs")
    return all_pages