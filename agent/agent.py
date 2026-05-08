import os
from dotenv import load_dotenv
from groq import Groq
from agent.tools import vector_search, math_explainer

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """Tu es un assistant pédagogique spécialisé dans les cours de 
Théorie des Langages & Automates et Mathématiques de l'Ingénieur.

Tu réponds UNIQUEMENT à partir du contexte fourni par les documents de cours.
Si l'information n'est pas dans le contexte, dis-le clairement.
Réponds en français. Sois précis et structuré.
Termine toujours par une ligne "Sources :" listant les fichiers utilisés."""


def decide_tool(question: str) -> str:
    """Simple rule-based tool selection."""
    math_keywords = ["formule", "calcul", "matrice", "vecteur", "déterminant",
                     "eigenvalue", "noyau", "image", "base", "dimension"]
    
    question_lower = question.lower()
    if any(kw in question_lower for kw in math_keywords):
        return "math_explainer"
    return "vector_search"


def run_agent(question: str) -> str:
    # Step 1: decide which tool to use
    tool = decide_tool(question)
    
    # Step 2: retrieve context
    if tool == "math_explainer":
        context = math_explainer(question)
    else:
        context = vector_search(question)

    # Step 3: build prompt with context
    user_prompt = f"""Contexte extrait des documents de cours :

{context}

Question : {question}"""

    # Step 4: call Groq LLM
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    print("=== RAG Study Assistant ===")
    print("Tapez 'quit' pour quitter.\n")

    while True:
        question = input("Question > ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break

        print("\nRecherche en cours...\n")
        answer = run_agent(question)
        print("--- Réponse ---")
        print(answer)
        print()