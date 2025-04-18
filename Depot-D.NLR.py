
# Extracting Text From PDF

import os      
import fitz    
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_text_from_pdfs(project_dataset):
    """Extracts text from all PDFs in a given folder."""
    all_text_chunks = []  

    for pdf_file in os.listdir(project_dataset): 
        if pdf_file.endswith(".pdf"):  
            pdf_path = os.path.join(project_dataset, pdf_file)  
            doc = fitz.open(pdf_path)  
            
            for page_num, page in enumerate(doc): 
                text = page.get_text("text")  
                if text.strip():  
                    all_text_chunks.append({  
                        "content": text,  
                        "metadata": {"source": pdf_file, "page": page_num + 1}  
                    })
    
    return all_text_chunks  


# Chunking Them Into Documents

from langchain_core.documents import Document

def process_pdf(pdf_path):
    """Extracts text from a PDF and splits it into smaller chunks with metadata."""
    pdf_texts = extract_text_from_pdfs(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    # all_chunks = []
    documents= []
    for text_entry in pdf_texts:
        
        chunks = text_splitter.split_text(text_entry["content"])
        
        pagenumber = text_entry["metadata"]["page"]
        metadata = text_entry["metadata"]

        for idx, chunk in enumerate(chunks):
            # print(chunk[:10])
            # print(metadata)
            # print(f"{pagenumber}-{idx}")

            doc = Document(
                page_content=chunk,
                metadata=metadata,
                id=f"{pagenumber}-{idx}"
                
            )

            # print(doc)
            documents.append(doc)
            # all_chunks.append({"content": chunk, "metadata": text_entry["metadata"]})

    return documents
    
documents = process_pdf("./")


# Embedding Function / Embedding Setup for the Chunks / Documents

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


### Setting up The ChromaDB Vector Store

from langchain_chroma import Chroma
vector_store = Chroma(
    collection_name="Depot-D.NLR_db",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)


# Add Documents to The Vector Store

from uuid import uuid4
uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)


# Retriever Setup / Retriever Function

retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10}
)


# LLM Model

from langchain_community.llms import Ollama
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Optional: for streaming output
callbacks = [StreamingStdOutCallbackHandler()]

llm = Ollama(
    model="llama3.1",  # this should match the model name shown in `ollama list`
    temperature=0.01,
    top_p=0.95,
    repeat_penalty=1.03,
    callbacks=callbacks,
)

# Test it
#response = llm.invoke("What is Retrieval Augmented Generation?")
#print(response)


# Response Cleaner Function

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


# Prompt Template

from langchain_core.prompts import PromptTemplate

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

question_answering_prompt = PromptTemplate(template=SYSTEM_TEMPLATE, input_variables=["question", "context"])


# THE RAG CHAIN


from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    context = "\n\n".join(doc.page_content for doc in docs)
    print(f"----n{context}\n----")
    return context

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | question_answering_prompt
    | llm
    | StrOutputParser()
)


# Qyerying

query = "What is the procedure for train accident investigation?"
response = rag_chain.invoke(query)


# Cleaning the LLM Response

cleaned = clean_response(response)
# Add metadata from the top 2-3 retrieved documents
docs = retriever.get_relevant_documents(query)
sources = []
for doc in docs:
    meta = doc.metadata
    sources.append(f"[Source: {meta.get('source', 'unknown')} | Page: {meta.get('page', '?')}]")
# Remove duplicates
sources = list(set(sources))
# Final formatted output
final_output = cleaned + "\n\n" + "\n".join(sources)
print(final_output)