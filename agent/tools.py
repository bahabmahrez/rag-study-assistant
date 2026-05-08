from sentence_transformers import SentenceTransformer
from ingestion.embedder import get_collection

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def vector_search(question: str, top_k: int = 3) -> str:
    """Search the course material and return relevant chunks."""
    model = get_model()
    collection = get_collection()

    embedding = model.encode(question).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)

    if not results["documents"][0]:
        return "Aucun résultat trouvé dans les documents."

    output = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        source = meta["source"]
        page = meta["page"]
        output.append(f"[Source: {source} | Page {page}]\n{doc}")

    return "\n\n---\n\n".join(output)


def math_explainer(expression: str) -> str:
    """Search for a math or formal notation concept in the course material."""
    query = f"définition formule explication {expression}"
    return vector_search(query, top_k=2)