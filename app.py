import streamlit as st
from agent.agent import run_agent
from ingestion.embedder import get_collection
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a, #111827);
        color: white;
    }

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* Title */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.3rem;
        color: white;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #cbd5e1;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }

    /* Cards */
    .custom-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 1.2rem;
        border-radius: 18px;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 18px;
        padding: 0.7rem;
        margin-bottom: 0.8rem;
        border: 1px solid rgba(255,255,255,0.05);
        background: rgba(255,255,255,0.03);
    }

    /* User message */
    [data-testid="stChatMessageContent"] p {
        font-size: 1rem;
        line-height: 1.6;
    }

    /* Radio buttons */
    div[role="radiogroup"] {
        background: rgba(255,255,255,0.05);
        padding: 0.6rem;
        border-radius: 14px;
    }

    /* Chat input */
    .stChatInputContainer {
        border-top: none;
        background: transparent;
    }

    /* Buttons / inputs */
    .stTextInput input {
        border-radius: 12px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: rgba(255,255,255,0.08);
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-title">📚 RAG Study Assistant</div>
    <div class="subtitle">
        Posez vos questions sur les cours de TLA et Mathématiques de l'Ingénieur
    </div>
    """,
    unsafe_allow_html=True
)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    mode = st.radio(
        "Choisir le mode",
        ["🔗 Pipeline (v1)", "🤖 Agent ReAct (v2)"]
    )

    st.markdown("---")

    if mode == "🔗 Pipeline (v1)":
        st.markdown("""
        <div class="custom-card">
        <h4>🔗 Pipeline classique</h4>
        <p>
        Une seule recherche documentaire suivie d'une réponse directe.
        </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="custom-card">
        <h4>🤖 Agent intelligent</h4>
        <p>
        L'agent décide dynamiquement quels outils utiliser et combien de recherches effectuer.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Powered by Groq • LangChain • ChromaDB")

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline v1
# ──────────────────────────────────────────────────────────────────────────────
def run_pipeline(question: str) -> str:
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    collection = get_collection()

    embedding = model.encode(question).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=3)

    context_parts = []
    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0]),
        start=1
    ):
        context_parts.append(
            f"[Extrait {i} | source={meta['source']} | page={meta['page']}]\n{doc}"
        )

    context = "\n\n---\n\n".join(context_parts)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un assistant pédagogique. "
                    "Réponds uniquement à partir du contexte fourni. "
                    "Réponds en français. "
                    "Termine par 'Sources :' avec les fichiers utilisés."
                )
            },
            {
                "role": "user",
                "content": f"Contexte :\n{context}\n\nQuestion : {question}"
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content


# ──────────────────────────────────────────────────────────────────────────────
# Chat State
# ──────────────────────────────────────────────────────────────────────────────
key = "messages_v1" if mode == "🔗 Pipeline (v1)" else "messages_v2"

if key not in st.session_state:
    st.session_state[key] = []

# ──────────────────────────────────────────────────────────────────────────────
# Chat History
# ──────────────────────────────────────────────────────────────────────────────
for message in st.session_state[key]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ──────────────────────────────────────────────────────────────────────────────
# User Input
# ──────────────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("💬 Posez votre question ici..."):

    st.session_state[key].append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("🔍 Recherche et génération de réponse..."):

            if mode == "🔗 Pipeline (v1)":
                answer = run_pipeline(prompt)
            else:
                answer = run_agent(prompt)

        st.markdown(answer)

    st.session_state[key].append({
        "role": "assistant",
        "content": answer
    })