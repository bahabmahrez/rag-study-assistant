import chromadb
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "study_materials"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def get_collection(persist_dir: str = "./chroma_db"):
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def embed_and_store(chunks: list[dict], persist_dir: str = "./chroma_db"):
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    collection = get_collection(persist_dir)

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [{
        "source": c["source"],
        "page": c["page"],
        "chunk_index": c["chunk_index"],
    } for c in chunks]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB")
    return collection