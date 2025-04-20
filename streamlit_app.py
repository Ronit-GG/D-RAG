import streamlit as st
import os
import time
from D_RAG_utils import (
    setup_vector_store, setup_llm, build_rag_chain,
    clean_response, format_rag_output, format_docs,
    process_pdf
)
from langchain_core.documents import Document
from io import StringIO
import datetime

st.set_page_config(page_title="Depot-D.NLR", layout="wide")


# --- Initialize Sidebar Toggle ---
if "show_sidebar" not in st.session_state:
    st.session_state.show_sidebar = True

# --- Inject CSS + HTML button for floating sidebar toggle ---
st.markdown(f"""
    <style>
        .sidebar-toggle {{
            position: fixed;
            top: 14px;
            left: 14px;
            z-index: 9999;
            background: #1f1f1f;
            color: #fff;
            border: none;
            padding: 4px 10px;
            border-radius: 5px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }}
    </style>
    <form action="" method="post">
        <button class="sidebar-toggle" name="toggle_sidebar" type="submit">
            {"<" if st.session_state.show_sidebar else ">"}
        </button>
    </form>
""", unsafe_allow_html=True)

# --- Toggle Logic ---
if "toggle_sidebar" in st.session_state.get("_form_data", {}):  # this catches Streamlit’s auto-form submission
    st.session_state.show_sidebar = not st.session_state.show_sidebar


if st.session_state.show_sidebar:
    with st.sidebar:
        st.title("📂 Depot-D.NLR")
        st.markdown("Built for Indian Railways document search.")

        st.markdown("### Upload PDFs")
        uploaded_files = st.file_uploader("Upload new PDFs", type="pdf", accept_multiple_files=True)

        if uploaded_files:
            os.makedirs("./pdfs", exist_ok=True)
            for file in uploaded_files:
                with open(os.path.join("./pdfs", file.name), "wb") as f:
                    f.write(file.read())
            st.success("PDFs uploaded!")

            if st.button("🔄 Rebuild Index"):
                with st.spinner("Indexing..."):
                    process_pdf("./pdfs")
                    st.success("Index rebuilt!")

        st.markdown("---")
        debug_mode = st.checkbox("🛠 Developer Debug Mode", value=st.session_state.get("debug_mode", False))
        st.session_state.debug_mode = debug_mode

        if st.button("🧹 Clear Chat"):
            st.session_state.messages = []

        if st.button("📥 Download Chat"):
            if "messages" in st.session_state and st.session_state.messages:
                chat_log = "\n\n".join(
                    f"User: {msg['content']}" if msg["role"] == "user"
                    else f"Assistant: {msg['content']}"
                    for msg in st.session_state.messages
                )
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"Depot-DNLR_Chat_{timestamp}.txt"
                st.download_button("Download Chat Log", chat_log, file_name=filename)
            else:
                st.warning("No chat to download.")
        st.markdown("---")
else:
    st.session_state.debug_mode = False



# --- Init ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vector_store()

if "llm" not in st.session_state:
    st.session_state.llm = setup_llm()

if "rag_chain" not in st.session_state:
    retriever = st.session_state.vectorstore.as_retriever(search_type="mmr")
    st.session_state.rag_chain = build_rag_chain(retriever, st.session_state.llm)

# --- Main Chat UI ---
st.title("🚆 Depot-D.NLR – Indian Railways Chatbot")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("NAMASTE, Ask me anything from the Railway documents!")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        start_time = time.time()

        docs = st.session_state.vectorstore.similarity_search(prompt, k=4)

        for chunk in st.session_state.rag_chain.stream(prompt):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")

        end_time = time.time()

        cleaned = clean_response(full_response)
        final_response = format_rag_output(cleaned, docs)

        response_placeholder.markdown(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})

        if debug_mode:
            with st.expander("🪛 Debug Info"):
                st.markdown(f"**Retrieved Chunks:**")
                for i, doc in enumerate(docs):
                    st.markdown(f"`[{i+1}]` {doc.metadata['source']} - Page {doc.metadata['page']}")
                    st.markdown(f"> {doc.page_content[:400]}...")

                st.markdown(f"**Prompt Sent to LLM:**\n\n```{prompt}```")
                st.markdown(f"**Latency:** `{end_time - start_time:.2f}s`")
