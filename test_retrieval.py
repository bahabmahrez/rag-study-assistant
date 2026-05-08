from ingestion.embedder import get_collection
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

def test_query(question: str, top_k: int = 3):
    model = SentenceTransformer(MODEL_NAME)
    collection = get_collection()

    embedding = model.encode(question).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)

    print(f"\nQuestion: {question}")
    print("=" * 60)
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"\n[{i+1}] {meta['source']} | page {meta['page']}")
        print(doc[:300])
        print("...")

if __name__ == "__main__":
    test_query("Qu'est-ce qu'un automate fini déterministe ?")
    test_query("Définition d'un espace vectoriel")