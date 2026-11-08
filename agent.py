"""Point 4 : pipeline RAG simple.  Point 6 : agent LangChain qui choisit ses outils de recherche."""
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import VectorStoreQuery

from ingestion import clip_text_embedding, store

# LLM local servi par Ollama. Version "instruct" de Qwen3 : répond directement, sans phase de réflexion.
# temperature=0 : réponses déterministes, collées au contexte.
llm = ChatOllama(model="qwen3:4b-instruct", temperature=0)

text_retriever = VectorStoreIndex.from_vector_store(store("text_chunks", 1024)).as_retriever(similarity_top_k=5)
image_store = store("image_clip", 512)

SYSTEM = """Tu es un assistant qui répond à des questions sur des documents techniques.
Réponds uniquement à partir du contexte fourni et cite les pages sous la forme (p. N).
Si la réponse n'est pas dans le contexte, réponds : "Je ne sais pas, l'information n'est pas dans le document."
Réponds en français."""


def _format(nodes) -> tuple[str, list[dict]]:
    """Transforme les nœuds retrouvés en (contexte texte pour le LLM, sources pour l'interface)."""
    context = "\n\n".join(f"[page {n.metadata['page']} | {n.metadata['type']}]\n{n.get_content()}" for n in nodes)
    sources = [{"page": n.metadata["page"], "type": n.metadata["type"],
                "image_path": n.metadata.get("image_path"), "text": n.get_content()} for n in nodes]
    return context, sources


# ---------- Point 4 : RAG simple (toujours la même chaîne : recherche -> prompt -> LLM) ----------
def rag(question: str) -> tuple[str, list[dict]]:
    context, sources = _format(text_retriever.retrieve(question))
    prompt = f"{SYSTEM}\n\n<context>\n{context}\n</context>\n\n<question>\n{question}\n</question>"
    return llm.invoke(prompt).content, sources


# ---------- Point 6 : outils que l'agent peut choisir d'appeler ----------
@tool(response_format="content_and_artifact")
def search_text(query: str):
    """Recherche dans le document les passages de texte, les tableaux et les descriptions de figures
    pertinents pour la requête. À utiliser pour toute question sur le contenu du document."""
    return _format(text_retriever.retrieve(query))


@tool(response_format="content_and_artifact")
def search_images(query: str):
    """Recherche les figures du document qui correspondent visuellement à la requête (CLIP).
    À utiliser quand on demande une figure, un schéma, une image ou une illustration.
    La requête doit être rédigée en anglais (CLIP ne comprend que l'anglais)."""
    result = image_store.query(VectorStoreQuery(query_embedding=clip_text_embedding(query), similarity_top_k=2))
    return _format(result.nodes)


AGENT_PROMPT = """Tu es un assistant qui répond à des questions sur des documents techniques indexés.
Tu ne connais pas le document : pour chaque nouvelle question, appelle d'abord un outil de recherche,
même si la conversation contient déjà des résultats (ils concernaient d'autres questions).
- search_text : pour le contenu (texte, chiffres, tableaux, descriptions de figures).
- search_images : quand on demande une figure, un schéma ou une image (requête en anglais).
Ensuite réponds uniquement à partir des résultats des outils, en citant les pages sous la forme (p. N).
Si les résultats ne contiennent pas la réponse, réponds : "Je ne sais pas, l'information n'est pas dans le document."
Réponds en français."""

@wrap_model_call
def search_first(request, handler):
    """Garde-fou (middleware) : au début de chaque tour, le modèle doit chercher avant de répondre.
    Les petits modèles sautent parfois la recherche quand la conversation est déjà longue."""
    response = handler(request)
    turn_start = isinstance(request.messages[-1], HumanMessage)  # aucun outil encore appelé pour cette question
    if turn_start and not response.result[-1].tool_calls:
        reminder = HumanMessage("Appelle d'abord un outil de recherche avant de répondre.")
        response = handler(request.override(messages=[*request.messages, reminder]))
    return response


# L'agent : le LLM raisonne, appelle les outils, lit leurs résultats et répond (boucle ReAct).
# Le checkpointer garde l'historique de chaque conversation (gestion du contexte).
agent = create_agent(model=llm, tools=[search_text, search_images], system_prompt=AGENT_PROMPT,
                     middleware=[search_first], checkpointer=InMemorySaver())


def ask(question: str, thread_id: str = "default") -> tuple[str, list[dict]]:
    result = agent.invoke({"messages": [HumanMessage(question)]}, config={"configurable": {"thread_id": thread_id}})
    messages = result["messages"]
    # Ne garder que les appels d'outils du tour actuel (après la dernière question)
    last_question = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    sources = [s for m in messages[last_question:] if isinstance(m, ToolMessage) for s in (m.artifact or [])]
    return messages[-1].content, sources


if __name__ == "__main__":
    while (q := input("Question (vide pour quitter) : ")):
        answer, sources = ask(q)
        print(answer, "\nSources :", sorted({(s["page"], s["type"]) for s in sources}), "\n")
