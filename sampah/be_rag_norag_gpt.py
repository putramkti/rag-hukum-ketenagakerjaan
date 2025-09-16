import os
from dotenv import load_dotenv, find_dotenv
import openai
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')  # Digunakan untuk embeddings dan juga GPT-4o

# Init Embeddings - tetap menggunakan OpenAI untuk embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Chroma DB init
persist_directory = "./chroma_db"

# FastAPI init
app = FastAPI(
    title="Ketenagakerjaan API Unified",
    description="API Terpadu untuk Sistem Question Answering (RAG dan Non-RAG)",
    version="0.1"
)

# Query model
class QueryRequest(BaseModel):
    query: str
    history: Optional[str] = ""
    mode: Optional[str] = "both"  # "rag", "no_rag", atau "both"

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

# Function untuk mendapatkan respons dengan RAG
async def get_rag_response(query: str, history: str):
    # Inisialisasi model GPT-4o dari OpenAI
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        max_tokens=1024,
        top_p=0.9
    )
    
    context_data = get_retriever_context(query)
    context = context_data["context"]
    chunks = context_data["chunks"]
    metadata = context_data["metadata"]
    
    print("CONTEXT RETRIEVED:", context)

    try:
        # Template untuk model GPT-4o
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
        """

        prompt = ChatPromptTemplate.from_template(template)

        chain = prompt | llm | StrOutputParser()

        result = chain.invoke({
            "context": context,
            "question": query,
            "history": history or ""
        })

        return {
            "response": result,
            "context_chunks": chunks,
            "metadata": metadata
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "response": f"Error saat memproses dengan RAG: {str(e)}",
            "context_chunks": [],
            "metadata": []
        }

# Function untuk mendapatkan respons tanpa RAG
async def get_no_rag_response(query: str, history: str):
    # Inisialisasi model GPT-4o dari OpenAI
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        max_tokens=1024,
        top_p=0.9
    )
    
    try:
        # Template untuk model tanpa RAG
        template = """
        Kamu adalah asisten hukum yang memiliki pengetahuan umum tentang regulasi ketenagakerjaan di Indonesia, khususnya UU Ketenagakerjaan No. 13 Tahun 2003.
        Tolong sertakan nomor pasal atau ayat .
        Berdasarkan pengetahuanmu, jawablah pertanyaan dengan relevan, akurat, dan jelas.
        
        CHAT HISTORY:
        {history}

        PERTANYAAN:
        {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        chain = prompt | llm | StrOutputParser()

        result = chain.invoke({
            "question": query,
            "history": history or ""
        })

        return {
            "response": result,
            "context_chunks": [],
            "metadata": []
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "response": f"Error saat memproses tanpa RAG: {str(e)}",
            "context_chunks": [],
            "metadata": []
        }

# Endpoint utama - terima parameter "mode" untuk menentukan jenis respons
@app.post("/ask")
async def ask_question(request: QueryRequest):
    print("QUERY:", request.query)
    print("HISTORY:", request.history)
    print("MODE:", request.mode)
    
    try:
        if request.mode == "rag":
            # Hanya RAG
            rag_result = await get_rag_response(request.query, request.history)
            return rag_result
        elif request.mode == "no_rag":
            # Hanya non-RAG
            no_rag_result = await get_no_rag_response(request.query, request.history)
            return no_rag_result
        else:
            # Mode default: keduanya, kembalikan dalam respons terpisah
            rag_result = await get_rag_response(request.query, request.history)
            no_rag_result = await get_no_rag_response(request.query, request.history)
            
            return {
                "rag": rag_result,
                "no_rag": no_rag_result
            }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint khusus RAG
@app.post("/ask/rag")
async def ask_rag(request: QueryRequest):
    request.mode = "rag"
    return await ask_question(request)

# Endpoint khusus non-RAG
@app.post("/ask/no_rag")
async def ask_no_rag(request: QueryRequest):
    request.mode = "no_rag"
    return await ask_question(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)