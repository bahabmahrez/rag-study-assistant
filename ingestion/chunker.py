from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_pages(pages: list[dict], chunk_size: int = 512, overlap: int = 50) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "],
    )

    chunks = []
    for page in pages:
        text = page["text"].strip()
        if not text:
            continue

        splits = splitter.split_text(text)

        for i, split in enumerate(splits):
            chunks.append({
                "text": split,
                "source": page["source"],
                "page": page["page"],
                "chunk_index": i,
                "id": f"{page['source']}_p{page['page']}_c{i}",
            })

    print(f"Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks