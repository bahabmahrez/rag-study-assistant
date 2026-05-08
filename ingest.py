from ingestion.extractor import extract_all_pdfs
from ingestion.chunker import chunk_pages
from ingestion.embedder import embed_and_store


def main():
    print("=== RAG Study Assistant — Ingestion Pipeline ===\n")

    pages = extract_all_pdfs("data/pdfs")
    if not pages:
        return

    chunks = chunk_pages(pages)
    embed_and_store(chunks)

    print("\nIngestion complete.")


if __name__ == "__main__":
    main()