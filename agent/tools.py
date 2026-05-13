from sentence_transformers import SentenceTransformer
from ingestion.embedder import get_collection
from langchain.tools import tool

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_model = None

TLA_KEYWORDS = [
    "automate", "afn", "afd", "langage", "arden", "regex",
    "expression régulière", "transition", "état", "alphabet",
    "grammaire", "mot", "epsilon", "thompson", "δ", "σ", "ε-fermeture",
    "fermeture", "déterministe", "non déterministe", "minimisation"
]

MATH_KEYWORDS = [
    "anneau", "groupe", "corps", "matrice", "vecteur", "espace vectoriel",
    "application linéaire", "déterminant", "noyau", "image", "base",
    "dimension", "sous-anneau", "sous-groupe", "isomorphisme", "homomorphisme",
    "centre", "inversible", "intègre", "valeur propre", "endomorphisme"
]


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def detect_subject(text: str) -> str:
    """Returns 'TLA', 'MATH', or None based on keyword matching."""
    text_lower = text.lower()
    if any(kw in text_lower for kw in TLA_KEYWORDS):
        return "TLA"
    if any(kw in text_lower for kw in MATH_KEYWORDS):
        return "MATH"
    return None


def query_collection(embedding, n_results: int, subject_filter: str = None):
    """Query ChromaDB with optional subject filter. Falls back to unfiltered if no results."""
    collection = get_collection()

    if subject_filter:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where={"source": {"$contains": subject_filter}}
        )
        # fallback to unfiltered if filter returns nothing
        if not results["documents"][0]:
            results = collection.query(
                query_embeddings=[embedding],
                n_results=n_results
            )
    else:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )

    return results


def format_results(results) -> str:
    """Format ChromaDB results into readable context string."""
    if not results["documents"][0]:
        return "Aucun résultat trouvé dans les documents."

    output = []
    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0]), start=1
    ):
        output.append(
            f"[Extrait {i} | source={meta['source']} | page={meta['page']}]\n{doc}"
        )

    return "\n\n---\n\n".join(output)


@tool
def vector_search(question: str) -> str:
    """Search the course material for relevant content. Use this for any question
    about TLA (automates, langages, expressions régulières) or Mathématiques
    (groupes, anneaux, espaces vectoriels, matrices, applications linéaires)."""
    model = get_model()
    embedding = model.encode(question).tolist()
    subject_filter = detect_subject(question)
    results = query_collection(embedding, n_results=3, subject_filter=subject_filter)
    return format_results(results)


@tool
def math_explainer(expression: str) -> str:
    """Use this specifically for formal math notation, formulas, or symbolic
    expressions — like δ, Σ, ε-fermeture, noyau, déterminant, valeurs propres,
    or any LaTeX-style expression."""
    query = f"définition formule démonstration {expression}"
    model = get_model()
    embedding = model.encode(query).tolist()

    # detect from expression first, default to MATH if unclear
    subject_filter = detect_subject(expression)
    if subject_filter is None:
        subject_filter = "MATH"

    results = query_collection(embedding, n_results=2, subject_filter=subject_filter)
    return format_results(results)