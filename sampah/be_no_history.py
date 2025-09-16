import os
from dotenv import load_dotenv, find_dotenv
import openai
from langchain_openai import OpenAIEmbeddings
# from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from langchain.retrievers import BM25Retriever, EnsembleRetriever

_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')

# Init Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
nomic_embeddings = OllamaEmbeddings(model="nomic-embed-text")
# Inisialisasi IndoBERT sebagai embedding
indobert_embeddings = HuggingFaceEmbeddings(
    model_name="indobenchmark/indobert-base-p1"
)

# Chroma DB init
persist_directory = "./chroma_db_indobert_gabungan_pasal"

# FastAPI init
app = FastAPI(title="Ketenagakerjaan API", version="0.1")

class QueryRequest(BaseModel):
    query: str
    mode: Optional[str] = "both"  # "rag", "no_rag", atau "both"

def get_retriever_context(input_query):
    try:
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=indobert_embeddings
        )
        documents = db.similarity_search(input_query, k=10)
        chunks = [doc.page_content for doc in documents] if documents else []
        context = "\n\n".join(chunks) if chunks else "Tidak ada dokumen relevan ditemukan."
        return {
            "context": context,
            "chunks": chunks,
            "metadata": [doc.metadata for doc in documents],
        }
    except Exception as e:
        return {
            "context": f"Error saat mengambil dokumen: {str(e)}",
            "chunks": [],
            "metadata": []
        }

async def get_rag_response(query: str):
    llm = OllamaLLM(model="llama3.2", temperature=0.3, verbose=True)
    context_data = get_retriever_context(query)
    context = context_data["context"]
    chunks = context_data["chunks"]
    metadata = context_data["metadata"]

    try:
        template = """<|system|>
            Kamu adalah asisten hukum yang ahli dalam regulasi ketenagakerjaan di Indonesia, khususnya Undang-Undang Nomor 13 Tahun 2003 tentang Ketenagakerjaan. 
            Tugasmu adalah menjawab pertanyaan pengguna dengan akurat, hanya berdasarkan konteks yang diberikan berupa kutipan resmi dari UU No. 13 Tahun 2003. 
            Jawaban harus:
            - Berdasarkan kutipan konteks yang tersedia, tanpa menambah atau mengarang informasi.
            - Menyebutkan pasal dan ayat secara eksplisit (jika disebutkan dalam konteks).
            - Menyatakan secara jujur jika jawaban tidak ditemukan dalam konteks yang diberikan.
            - Disampaikan secara jelas, dan objektif, sesuai gaya komunikasi hukum.


            Perhatikan bahwa:
            - Ayat ditulis seperti: `ayat (1)`, `ayat (2)`, dan seterusnya.
            - Setiap pasal bisa diikuti oleh beberapa ayat yang dijelaskan satu per satu.
            - Jangan buat interpretasi di luar isi kutipan yang diberikan.
            - Jika kutipan tidak mencakup ayat atau pasal yang relevan, jawab "Informasi tidak tersedia dalam konteks."

            <|user|>
            Berikut adalah konteks dari dokumen hukum yang relevan (kutipan dari UU No. 13 Tahun 2003):

            {context}

            Pertanyaan:
            {question}
            <|assistant|>

        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"context": context, "question": query})
        return {
            "response": result,
            "context_chunks": chunks,
            "metadata": metadata
        }
    except Exception as e:
        return {
            "response": f"Error saat memproses dengan RAG: {str(e)}",
            "context_chunks": [],
            "metadata": []
        }

async def get_no_rag_response(query: str):
    llm = OllamaLLM(model="llama3.2", temperature=0.3, verbose=True)
    try:
        template = """<|begin_of_text|>
        <|system|>
        Kamu adalah asisten hukum profesional yang hanya menjawab pertanyaan berdasarkan peraturan resmi ketenagakerjaan di Indonesia, khususnya UU No. 13 Tahun 2003. Jawablah dengan akurat, berdasarkan pasal dan ayat. Jika tidak tahu, katakan tidak tahu.
        <|user|>
        Pertanyaan pengguna:
        {question}
        <|assistant|>
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"question": query})
        return {
            "response": result,
            "context_chunks": [],
            "metadata": []
        }
    except Exception as e:
        return {
            "response": f"Error saat memproses tanpa RAG: {str(e)}",
            "context_chunks": [],
            "metadata": []
        }

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        if request.mode == "rag":
            return await get_rag_response(request.query)
        elif request.mode == "no_rag":
            return await get_no_rag_response(request.query)
        else:
            return {
                "rag": await get_rag_response(request.query),
                "no_rag": await get_no_rag_response(request.query)
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
