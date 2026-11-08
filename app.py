"""Point 5 : interface Streamlit (chargement du PDF, questions, réponses avec sources)."""
import uuid

import streamlit as st

from agent import ask, rag
from ingestion import MEDIA_DIR, ingest

st.set_page_config(page_title="RAG multimodal agentique", layout="wide")
st.title("RAG multimodal agentique")

# st.session_state survit aux ré-exécutions du script à chaque interaction
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())  # une conversation = une mémoire d'agent
    st.session_state.history = []

with st.sidebar:
    pdf = st.file_uploader("Charger un PDF", type="pdf")
    if pdf and st.button("Indexer le PDF"):
        MEDIA_DIR.mkdir(exist_ok=True)
        path = MEDIA_DIR / pdf.name
        path.write_bytes(pdf.getvalue())
        with st.status("Indexation en cours…", expanded=True) as status:
            ingest(str(path), progress=st.write)
            status.update(label="PDF indexé", state="complete")
    mode = st.radio("Mode", ["Agent (LangChain)", "RAG simple"])


def show(answer: str, sources: list[dict]) -> None:
    st.markdown(answer)
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for s in sources:
                st.caption(f"Page {s['page']} · {s['type']}")
                if s["image_path"]:
                    st.image(s["image_path"], width=500)
                else:
                    st.text(s["text"][:600])


for role, answer, sources in st.session_state.history:
    with st.chat_message(role):
        show(answer, sources)

if question := st.chat_input("Pose une question sur le document"):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"), st.spinner("Recherche et génération…"):
        if mode == "RAG simple":
            answer, sources = rag(question)
        else:
            answer, sources = ask(question, st.session_state.thread_id)
        show(answer, sources)
    st.session_state.history += [("user", question, []), ("assistant", answer, sources)]
