import json
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="AI Document Chat",
    page_icon="🧠",
    layout="centered",
)

st.title("AI Document Chat")
st.caption("Upload documents, then ask questions. Powered by RAG + OpenAI / Claude.")

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")

    provider = st.radio(
        "LLM Provider",
        options=["claude", "openai"],
        format_func=lambda x: "OpenAI GPT-4o-mini" if x == "openai" else "Anthropic Claude",
    )

    k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=5)

    st.divider()

    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Ingest Documents", disabled=not uploaded_files, type="primary"):
        docs_dir = "documents"
        os.makedirs(docs_dir, exist_ok=True)

        for f in uploaded_files:
            with open(os.path.join(docs_dir, f.name), "wb") as out:
                out.write(f.read())

        with st.spinner("Embedding documents..."):
            try:
                resp = requests.post(f"{API_URL}/ingest", timeout=120)
                resp.raise_for_status()
                data = resp.json()
                st.success(f"Ingested {data['chunks_created']} chunks!")
            except Exception as e:
                st.error(f"Ingest failed: {e}")

    st.divider()

    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        db_ready = health.get("vector_db_ready", False)
        st.status(
            "Vector DB ready" if db_ready else "No documents ingested yet",
            state="complete" if db_ready else "error",
        )
    except Exception:
        st.warning("API unreachable — is uvicorn running?")


# --- Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            with requests.post(
                f"{API_URL}/chat",
                json={"question": prompt, "provider": provider, "k": k},
                stream=True,
                timeout=60,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8")
                    if decoded == "data: [DONE]":
                        break
                    if decoded.startswith("data: "):
                        payload = json.loads(decoded[6:])
                        full_response += payload.get("text", "")
                        placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"Error: {e}"
            placeholder.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
