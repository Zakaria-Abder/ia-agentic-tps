#!/usr/bin/env python
# coding: utf-8

"""
FinanceAdvisor.py
Projet : Assistant RAG Finance avec LangGraph
Compatible avec LangGraph Studio
"""

# =====================================================
# IMPORTS
# =====================================================

from typing import Annotated

from typing_extensions import TypedDict

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_ollama import ChatOllama

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from langchain_core.tools import tool

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.graph.message import add_messages

from langgraph.prebuilt import (
    ToolNode,
    tools_condition,
)

print("=" * 60)
print("Finance Advisor")
print("=" * 60)


# =====================================================
# CHARGEMENT DES PDF
# =====================================================

loader = PyPDFDirectoryLoader("data")

documents = loader.load()

print(f"{len(documents)} pages chargées.")


# =====================================================
# DECOUPAGE
# =====================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"{len(chunks)} chunks créés.")


# =====================================================
# EMBEDDINGS
# =====================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings chargés.")


# =====================================================
# VECTOR STORE
# =====================================================

vector_store = InMemoryVectorStore(
    embeddings
)

vector_store.add_documents(
    documents=chunks
)

print("Vector Store créé.")


# =====================================================
# MODELE
# =====================================================

model = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)

print("LLM chargé.")

# =====================================================
# TOOLS
# =====================================================

@tool
def search_finance(question: str) -> str:
    """
    Recherche les informations pertinentes dans les documents de finance.
    """

    docs = vector_store.similarity_search(
        question,
        k=3
    )

    if len(docs) == 0:
        return "Aucun document pertinent trouvé."

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


@tool
def answer_finance(question: str) -> str:
    """
    Génère une réponse à partir des documents retrouvés.
    """

    context = search_finance.invoke(question)

    prompt = f"""
Tu es un assistant spécialisé en finance.

Tu dois répondre uniquement à partir du contexte fourni.

Contexte :

{context}

Question :

{question}

Réponse :
"""

    response = model.invoke(prompt)

    return response.content


# =====================================================
# LISTE DES TOOLS
# =====================================================

tools = [
    search_finance,
    answer_finance
]

print("Tools créés.")


# =====================================================
# MODELE + TOOLS
# =====================================================

model_with_tools = model.bind_tools(
    tools
)

print("Le modèle est maintenant connecté aux Tools.")

# =====================================================
# STATE
# =====================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# =====================================================
# ASSISTANT NODE
# =====================================================

def assistant(state: AgentState):

    response = model_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [
            response
        ]
    }


# =====================================================
# TOOL NODE
# =====================================================

tool_node = ToolNode(
    [search_finance]
)


# =====================================================
# CONSTRUCTION DU GRAPHE
# =====================================================

builder = StateGraph(AgentState)

builder.add_node(
    "assistant",
    assistant
)

builder.add_node(
    "tools",
    tool_node
)


# =====================================================
# EDGES
# =====================================================

builder.add_edge(
    START,
    "assistant"
)

builder.add_conditional_edges(
    "assistant",
    tools_condition
)

builder.add_edge(
    "tools",
    "assistant"
)

# =====================================================
# COMPILATION DU GRAPHE
# =====================================================

builder.add_edge(
    "assistant",
    END
)

graph = builder.compile()

print("Graph LangGraph compilé avec succès.")


# =====================================================
# TEST LOCAL
# =====================================================

if __name__ == "__main__":

    config = {
        "configurable": {
            "thread_id": "finance-session"
        }
    }

    question = HumanMessage(
        content="Qu'est-ce qu'une action ?"
    )

    response = graph.invoke(
        {
            "messages": [question]
        },
        config=config
    )

    print("\n==============================")
    print("Réponse de l'agent")
    print("==============================\n")

    print(response["messages"][-1].content)
