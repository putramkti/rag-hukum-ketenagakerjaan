# be.py
import os
from dotenv import load_dotenv, find_dotenv
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')  # Tetap digunakan untuk embeddings

# Init Embeddings - tetap menggunakan OpenAI untuk embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Alternatif: Gunakan embeddings lokal seperti BGE
# from langchain_community.embeddings import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

# Chroma DB init
persist_directory = "./chroma_db"

# FastAPI init
app = FastAPI(
    title="Ketenagakerjaan API",
    description="API untuk Sistem Question Answering Regulasi Ketenagakerjaan di Indonesia",
    version="0.1"
)

# Query model
class QueryRequest(BaseModel):
    query: str
    history: Optional[str] = ""

def format_docs_with_references(docs):
    """Format documents with their references for context"""
    formatted = ""
    for doc in docs:
        ref = doc.metadata.get("full_reference", "Tidak diketahui")
        formatted += f"[{ref}]\n{doc.page_content}\n\n"
    return formatted


# Define retriever as a tool
def get_retriever_context(input_query):
    search_query = input_query
    print("SEARCH QUERY:", search_query)
    try:
        # Inisialisasi db dan retriever
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        # Cari dokumen yang relevan
        documents = db.similarity_search(search_query, k=4)
        # Return both raw documents and joined context
        context_chunks = [f"{doc.metadata.get('full_reference', 'Tidak diketahui')}\n{doc.page_content}" for doc in documents] if documents else []
        context = "\n\n".join(context_chunks) if context_chunks else "Tidak ada dokumen relevan ditemukan."
        

        
        return {
            "context": context,
            "chunks": context_chunks,
            "metadata": [doc.metadata for doc in documents],
        }
    except Exception as e:
        return {
            "context": f"Error saat mengambil dokumen: {str(e)}",
            "chunks": [],
            "metadata": []
        }

# Endpoint
@app.post("/ask")
async def ask_question(request: QueryRequest):
    # Inisialisasi model Ollama dengan Mistral 7B
    llm = Ollama(
        model="llama3.2",  # Gunakan model Mistral 7B
        temperature=0.3,
        verbose=True
    )
    
    print("QUERY:", request.query)
    
    context_data = get_retriever_context(request.query)
    context = context_data["context"]
    chunks = context_data["chunks"]
    metadata = context_data["metadata"]
    
    print("CONTEXT RETRIEVED:", context)

    try:
        # Template dengan format yang lebih sederhana untuk model Mistral 7B
        template = """
        Kamu adalah asisten hukum yang ahli tentang regulasi ketenagakerjaan di Indonesia, khususnya UU Ketenagakerjaan No. 13 Tahun 2003.
        Berdasarkan konteks berikut, jawablah pertanyaan dengan relevan, akurat, dan jelas.
        Sertakan nomor pasal dan ayat jika tersedia dalam konteks.
        Jika ada informasi yang tidak relevan, abaikan informasi tersebut.
        Informasi di konteks merupakan bagian dari UU No. 13 Tahun 2003.
        Jika informasi tidak tersedia dalam konteks, cukup jawab bahwa kamu tidak menemukan informasi yang relevan dalam regulasi.
        
        KONTEKS:
        {context}

        CHAT HISTORY:
        {history}

        PERTANYAAN:
        {question}

        JAWABAN:
        """

        prompt = ChatPromptTemplate.from_template(template)

        chain = prompt | llm | StrOutputParser()

        result = chain.invoke({
            "context": context,
            "question": request.query,
            "history": request.history or ""
        })

        return {
            "response": result,
            "context_chunks": chunks,
            "metadata": metadata
        }

    except Exception as e:
        import traceback
        traceback.print_exc()  # ini akan mencetak error ke terminal
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)