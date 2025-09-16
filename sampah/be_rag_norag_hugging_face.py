import os
from dotenv import load_dotenv, find_dotenv
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')  # Tetap digunakan untuk embeddings
huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')  
# Tambahkan API key untuk Hugging Face
print("HUGGINGFACE_API_KEY:", huggingface_api_key)
print("OPENAI_API_KEY:", openai.api_key)
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
    # Inisialisasi model Llama 3.2 dari HuggingFace
    llm = HuggingFaceEndpoint(
        # provider="together",
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        huggingfacehub_api_token=huggingface_api_key,
        temperature=0.3,
        max_new_tokens=1024,
        # top_p=0.9
    )

    # hf_pipeline = pipeline(
    #     task="text-generation",
    #     model="meta-llama/Llama-3.2-3B-Instruct",
    #     tokenizer="meta-llama/Llama-3.2-3B-Instruct",
    #     max_new_tokens=1024,
    #     temperature=0.3,
    #     top_p=0.9,
    #     do_sample=True
    # )
    # llm = HuggingFacePipeline(pipeline=hf_pipeline)
    
    # llm = HuggingFacePipeline.from_model_id(
    #     model_id="meta-llama/Llama-3.2-3B-Instruct",
    #     task="text-generation",
    #     huggingfacehub_api_token=huggingface_api_key,
    #     temperature=0.3,
    # )
    context_data = get_retriever_context(query)
    context = context_data["context"]
    chunks = context_data["chunks"]
    metadata = context_data["metadata"]
    
    print("CONTEXT RETRIEVED:", context)

    try:
        # Template dengan format yang lebih sederhana untuk model Llama 3.2
        template = """
            <|begin_of_text|><|system|>
            Kamu adalah asisten hukum yang ahli tentang regulasi ketenagakerjaan di Indonesia, khususnya UU Ketenagakerjaan No. 13 Tahun 2003.
            Berdasarkan konteks berikut, jawablah pertanyaan dengan relevan, akurat, dan jelas.
            Sertakan nomor pasal dan ayat jika tersedia dalam konteks.
            Jika ada informasi yang tidak relevan, abaikan informasi tersebut.
            Informasi di konteks merupakan bagian dari UU No. 13 Tahun 2003.
            Jika informasi tidak tersedia dalam konteks, cukup jawab bahwa kamu tidak menemukan informasi yang relevan dalam regulasi.
            </|system|>

            <|user|>
            KONTEKS:
            {context}
            
            CHAT HISTORY:
            {history}

            PERTANYAAN:
            {question}
            </|user|>

            <|assistant|>
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
    # Inisialisasi model Llama 3.2 dari HuggingFace
    llm = HuggingFaceEndpoint(
        # provider="together",
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        huggingfacehub_api_token=huggingface_api_key,
        temperature=0.3,
        max_new_tokens=1024,
        # top_p=0.9,
    )
    # hf_pipeline = pipeline(
    #     task="text-generation",
    #     model="meta-llama/Llama-3.2-3B-Instruct",
    #     tokenizer="meta-llama/Llama-3.2-3B-Instruct",
    #     max_new_tokens=1024,
    #     temperature=0.3,
    #     top_p=0.9,
    #     do_sample=True
    # )
    # llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # llm = HuggingFacePipeline.from_model_id(
    #     model_id="meta-llama/Llama-3.2-3B-Instruct",
    #     task="text-generation",
    #     huggingfacehub_api_token=huggingface_api_key,
    #     temperature=0.3,
    # )
    
    try:
        # Template untuk model tanpa RAG
        template = """
        <|begin_of_text|><|system|>
        Kamu adalah asisten hukum yang memiliki pengetahuan umum tentang regulasi ketenagakerjaan di Indonesia, khususnya UU Ketenagakerjaan No. 13 Tahun 2003.
        Berdasarkan pengetahuanmu, jawablah pertanyaan dengan relevan, akurat, dan jelas.
        Jangan mengklaim pengetahuan spesifik tentang nomor pasal atau ayat kecuali kamu sangat yakin.
        Jika kamu tidak yakin tentang suatu informasi, sampaikan bahwa kamu tidak memiliki informasi yang cukup.
        </|system|>
        
        <|user|>
        CHAT HISTORY:
        {history}

        PERTANYAAN:
        {question}
        </|user|>

        <|assistant|>
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
    uvicorn.run(app, host="0.0.0.0", port=8000)