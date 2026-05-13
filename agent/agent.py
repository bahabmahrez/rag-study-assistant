import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain.agents import AgentExecutor
from langchain.agents.tool_calling_agent.base import create_tool_calling_agent

from langchain_core.prompts import ChatPromptTemplate

from agent.tools import vector_search, math_explainer

load_dotenv()

SYSTEM_PROMPT = """Tu es un assistant pédagogique spécialisé dans deux matières :
- Théorie des Langages & Automates (TLA) : automates finis, langages réguliers, 
  expressions régulières, minimisation, ε-transitions
- Mathématiques de l'Ingénieur : groupes, anneaux, espaces vectoriels, 
  matrices, applications linéaires

Tu as accès à l'outil vector_search pour chercher du contenu dans les documents de cours.

Règles importantes :
- Utilise TOUJOURS vector_search avant de répondre — ne réponds jamais de mémoire
- Si une première recherche ne suffit pas, fais une deuxième recherche avec des termes différents
- Si l'information n'est pas dans les documents, dis-le clairement
- Réponds en français, de manière structurée et précise
- Termine toujours par "Sources :" en listant les fichiers utilisés"""


def build_agent():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    tools = [vector_search]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,          # shows the agent's reasoning steps
        max_iterations=4,      # prevents infinite loops
        handle_parsing_errors=True,
    )


_agent_executor = None


def run_agent(question: str) -> str:
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = build_agent()

    result = _agent_executor.invoke({"input": question})
    return result["output"]


if __name__ == "__main__":
    print("=== RAG Study Assistant (Agentic) ===")
    print("Tapez 'quit' pour quitter.\n")

    while True:
        question = input("Question > ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break

        print()
        answer = run_agent(question)
        print("\n--- Réponse finale ---")
        print(answer)
        print()