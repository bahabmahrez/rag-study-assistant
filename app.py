import streamlit as st
from agent.agent import run_agent
from ingestion.embedder import get_collection
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="RAG Study Assistant",
    page_icon="📚",
    layout="centered"
)

st.title("📚 RAG Study Assistant")
st.caption("Posez vos questions sur les cours de TLA et Mathématiques de l'Ingénieur.")

# ── Mode selector ────────────────────────────────────────────────────────────
mode = st.radio(
    "Mode",
    ["🔗 Pipeline (v1)", "🤖 Agent ReAct (v2)"],
    horizontal=True
)

if mode == "🔗 Pipeline (v1)":
    st.info("Version initiale — pipeline fixe : une question → une recherche → une réponse.")
else:
    st.success("Version améliorée — agent ReAct : le LLM décide quels outils appeler et combien de fois.")

st.divider()


# ── Pipeline v1 (original behavior) ─────────────────────────────────────────
def run_pipeline(question: str) -> str:
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    collection = get_collection()

    embedding = model.encode(question).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=3)

    context_parts = []
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), start=1):
        context_parts.append(f"[Extrait {i} | source={meta['source']} | page={meta['page']}]\n{doc}")
    context = "\n\n---\n\n".join(context_parts)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Tu es un assistant pédagogique. Réponds uniquement à partir du contexte fourni. Réponds en français. Termine par 'Sources :' avec les fichiers utilisés."},
            {"role": "user", "content": f"Contexte :\n{context}\n\nQuestion : {question}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


# ── Chat state per mode ───────────────────────────────────────────────────────
key = "messages_v1" if mode == "🔗 Pipeline (v1)" else "messages_v2"
if key not in st.session_state:
    st.session_state[key] = []

for message in st.session_state[key]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Posez votre question ici..."):
    st.session_state[key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            if mode == "🔗 Pipeline (v1)":
                answer = run_pipeline(prompt)
            else:
                answer = run_agent(prompt)
        st.markdown(answer)

    st.session_state[key].append({"role": "assistant", "content": answer})