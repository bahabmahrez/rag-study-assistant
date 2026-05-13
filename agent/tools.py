from sentence_transformers import SentenceTransformer
from ingestion.embedder import get_collection
from langchain.tools import tool

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


@tool
def vector_search(question: str) -> str:
    """Search the course material for relevant content. Use this for any question
    about TLA (automates, langages, expressions régulières) or Mathématiques
    (groupes, anneaux, espaces vectoriels, matrices, applications linéaires)."""
    model = get_model()
    collection = get_collection()

    embedding = model.encode(question).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=3)

    if not results["documents"][0]:
        return "Aucun résultat trouvé dans les documents."

    output = []
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), start=1):
        source = meta["source"]
        page = meta["page"]
        output.append(f"[Extrait {i} | source={source} | page={page}]\n{doc}")

    return "\n\n---\n\n".join(output)


@tool
def math_explainer(expression: str) -> str:
    """Use this specifically for formal math notation, formulas, or symbolic
    expressions that need to be looked up — like δ, Σ, ε-fermeture, noyau,
    déterminant, valeurs propres, or any LaTeX-style expression."""
    query = f"définition formule démonstration {expression}"
    model = get_model()
    collection = get_collection()

    embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=2)

    if not results["documents"][0]:
        return "Aucun résultat trouvé pour cette notation."

    output = []
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), start=1):
        source = meta["source"]
        page = meta["page"]
        output.append(f"[Extrait {i} | source={source} | page={page}]\n{doc}")

    return "\n\n---\n\n".join(output)