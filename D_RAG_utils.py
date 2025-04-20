import os
import fitz
from uuid import uuid4

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def extract_text_from_pdfs(folder_path):
    all_text_chunks = []

    for pdf_file in os.listdir(folder_path):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, pdf_file)
            doc = fitz.open(pdf_path)

            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    all_text_chunks.append({
                        "content": text,
                        "metadata": {"source": pdf_file, "page": page_num + 1}
                    })

    return all_text_chunks


def process_pdf(pdf_folder):
    pdf_texts = extract_text_from_pdfs(pdf_folder)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = []

    for text_entry in pdf_texts:
        chunks = text_splitter.split_text(text_entry["content"])
        pagenumber = text_entry["metadata"]["page"]
        metadata = text_entry["metadata"]

        for idx, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata=metadata,
                id=f"{pagenumber}-{idx}"
            )
            documents.append(doc)

    return documents


def setup_vector_store(pdf_folder="./pdfs"):
    documents = process_pdf(pdf_folder)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="Depot-D.NLR_db",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )
    uuids = [str(uuid4()) for _ in range(len(documents))]
    
    if not vector_store.get()["documents"]:  # add only if empty
        vector_store.add_documents(documents=documents, ids=uuids)

    return vector_store


def setup_llm():
    callbacks = [StreamingStdOutCallbackHandler()]
    return Ollama(
        model="llama3.1",
        temperature=0.01,
        top_p=0.95,
        repeat_penalty=1.03,
        callbacks=callbacks,
    )


SYSTEM_TEMPLATE = """
You are an assistant specialized in Indian Railways documentation. 
Answer the following question using only the provided context. 
Be concise, skip generic definitions, and prioritize information relevant to Indian Railways.

If the answer involves categories, list them clearly in bullet points or numbered format.
Always include source metadata like PDF name and page number if available.

Question: {question}

Context:
{context}
"""

question_answering_prompt = PromptTemplate(
    template=SYSTEM_TEMPLATE,
    input_variables=["question", "context"]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever, llm):
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | question_answering_prompt
        | llm
        | StrOutputParser()
    )


def clean_response(text):
    text = text.strip()
    if text.lower().startswith("train accident is"):
        text = text[text.find('.') + 1:].strip()

    keywords = ["include", "types of", "are as follows"]
    for word in keywords:
        if word in text.lower():
            text = text.replace("•", "-")
            break

    return text

def format_rag_output(answer: str, docs: list) -> str:
    # Deduplicate source references
    seen_sources = set()
    formatted_sources = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        citation = f"[Source: {source} | Page: {page}]"

        if citation not in seen_sources:
            formatted_sources.append(citation)
            seen_sources.add(citation)

    # Format sources block
    sources_text = "\n".join(formatted_sources)

    return f"""{answer.strip()}

**📁 Sources**
{sources_text}"""
