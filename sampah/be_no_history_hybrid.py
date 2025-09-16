import os
from dotenv import load_dotenv, find_dotenv
import openai
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import TFIDFRetriever
from langchain_core.documents import Document
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional



_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')

# Init Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# nomic_embeddings = OllamaEmbeddings(model="nomic-embed-text")
# indobert_embeddings = HuggingFaceEmbeddings(model_name="indobenchmark/indobert-base-p1")

# Ubah default param
DEFAULT_PERSIST_DIR = "./chroma_db_openai_gabungan_pasal_cosine"

app = FastAPI(title="Ketenagakerjaan API", version="0.1")

class QueryRequest(BaseModel):
    query: str
    mode: Optional[str] = "both"  # "rag", "no_rag", atau "both"

def build_hybrid_retriever(
    embedding_model,
    tfidf_k: int = 2,
    embed_k: int = 2,
    ensemble_k: int = 3,
    weights: list[float] = [0.7, 0.3],
    persist_directory: str = DEFAULT_PERSIST_DIR
):
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    raw_docs = db.get(include=["documents", "metadatas"])
    documents = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(raw_docs["documents"], raw_docs["metadatas"])
    ]

    tfidf_retriever = TFIDFRetriever.from_documents(documents)
    tfidf_retriever.k = tfidf_k

    embedding_retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": embed_k}
    )

    hybrid_retriever = EnsembleRetriever(
        retrievers=[embedding_retriever, tfidf_retriever],
        weights=weights,
        k=1
    )

    return hybrid_retriever

# def get_retriever_context(input_query):
#     try:
#         retriever = build_hybrid_retriever(embedding_model=embeddings,persist_directory="./chroma_db_openai_gabungan_pasal_cosine")
#         documents = retriever.invoke(input_query)
#         chunks = [doc.page_content for doc in documents] if documents else []
#         context = "\n\n".join(chunks) if chunks else "Tidak ada dokumen relevan ditemukan."
#         return {
#             "context": context,
#             "chunks": chunks,
#             "metadata": [doc.metadata for doc in documents],
#         }
#     except Exception as e:
#         return {
#             "context": f"Error saat mengambil dokumen: {str(e)}",
#             "chunks": [],
#             "metadata": []
#         }

def get_retriever_context(input_query):
    try:
        retriever = build_hybrid_retriever(embedding_model=embeddings, persist_directory="./chroma_db_openai_gabungan_pasal_cosine")

        # Ambil hasil dari masing-masing retriever manual
        tfidf_results = retriever.retrievers[1].invoke(input_query)
        embed_results = retriever.retrievers[0].invoke(input_query)
        final_results = retriever.invoke(input_query)

        tfidf_chunks = [doc.page_content for doc in tfidf_results]
        embed_chunks = [doc.page_content for doc in embed_results]
        final_chunks = [doc.page_content for doc in final_results]

        # print(f"TF-IDF Results: {tfidf_chunks} chunks")
        # print(f"Embedding Results: {embed_chunks} chunks")
        # print(f"\n\n".join(final_chunks))  # Print only the first 500 characters of final chunks

        return {
            "context": "\n\n".join(final_chunks),
            "chunks": final_chunks,
            "metadata": [doc.metadata for doc in final_results],
            "tfidf_chunks": tfidf_chunks,
            "embed_chunks": embed_chunks,
        }
    except Exception as e:
        return {
            "context": f"Error saat mengambil dokumen: {str(e)}",
            "chunks": [],
            "metadata": [],
            "tfidf_chunks": [],
            "embed_chunks": [],
        }

def get_rag_response(query: str):
    llm = OllamaLLM(
        model="qwen3:8b", 
        temperature=0, 
        verbose=True
    )

    context_data = get_retriever_context(query.lower())
    context = context_data["context"]
    chunks = context_data["chunks"]
    metadata = context_data["metadata"]

    print(f"Context: {context[:500]}...")  

    try:
        template = """<|im_start|>system
            Kamu adalah asisten hukum spesialis ketenagakerjaan di Indonesia. Tugasmu adalah menjawab pertanyaan pengguna secara singkat dan hanya berdasarkan kutipan resmi dari Undang-Undang yang tersedia dalam konteks, yaitu UU No. 13 Tahun 2003 tentang Ketenagakerjaan dan perubahannya, termasuk UU No. 6 Tahun 2023 (Cipta Kerja).

            Aturan ketat dalam menjawab:
            - Jawab secara lengkap, fokus pada pasal dan ayat yang relevan.
            - Jawaban hanya berdasarkan kutipan yang tersedia di konteks. 
            - Setiap penjelasan harus menyebutkan Pasal dan Ayat secara eksplisit.
            - Jika pasal telah diubah/dihapus/diganti, nyatakan secara eksplisit, misalnya: "Pasal 151 telah diubah oleh UU No. 6 Tahun 2023."
            - Jangan menambahkan interpretasi, penjelasan tambahan, atau pendapat pribadi.
            - Jangan menggunakan pengetahuan di luar konteks. Tidak boleh berasumsi.
            - Jika tidak ditemukan jawaban dalam konteks, jawab: "Informasi tidak tersedia dalam konteks."

            Format konteks:
            - Dimulai dengan nama Bab, contoh: Bab IX Hubungan Kerja.
            - Diikuti isi pasal dan ayat: (1), (2), dst.
            - Perubahan ditandai secara eksplisit, misalnya:
                - Pasal 151 (diubah oleh UU No. 6 Tahun 2023)
                - Pasal 151A (ditambahkan oleh UU No. 6 Tahun 2023)
                - Pasal 152 (dihapus oleh UU No. 6 Tahun 2023)

            Tujuan kamu adalah memberikan jawaban lengkap, eksplisit, dan hanya dari pasal/ayat yang relevan.
            <|im_end|>
            <|im_start|>user
            Berikut adalah konteks dari dokumen hukum yang relevan (kutipan dari UU No. 13 Tahun 2003 dan perubahannya):

            {context}

            Pertanyaan:
            (Jawab hanya jika ditemukan kutipan pasal yang relevan dalam konteks. Jangan tambahkan penjelasan.)
            {question}
            <|im_end|>
            <|im_start|>assistant
            /no_think
            """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"context": context, "question": query})

        return {
            "response": result,
            "context_chunks": chunks,
            "metadata": metadata,
            "tfidf_chunks": context_data["tfidf_chunks"],
            "embed_chunks": context_data["embed_chunks"]
        }

    except Exception as e:
        return {
            "response": f"Error saat memproses dengan RAG: {str(e)}",
            "context_chunks": [],
            "metadata": [],
            "tfidf_chunks": [],
            "embed_chunks": []
        }

# async def get_rag_response(query: str):
#     llm = OllamaLLM(
#             model="qwen3:8b", 
#             temperature=0, 
#             verbose=True)

#     context_data = get_retriever_context(query)
#     context = context_data["context"]
#     chunks = context_data["chunks"]
#     metadata = context_data["metadata"]

#     try:
#         template = """<|system|>
#             Kamu adalah asisten hukum yang ahli dalam regulasi ketenagakerjaan di Indonesia, khususnya Undang-Undang Nomor 13 Tahun 2003 tentang Ketenagakerjaan beserta perubahannya, termasuk dari UU No. 6 Tahun 2023 (Cipta Kerja). 
#             Tugasmu adalah menjawab pertanyaan pengguna secara akurat dan objektif, berdasarkan kutipan resmi dari Undang-Undang yang tersedia dalam konteks.

#             Aturan penjawaban:
#             - Jawab secara singkat, fokus pada pasal dan ayat yang relevan.
#             - jawab dengan format yang jelas, sebutkan pasal dan ayatnya.
#             - Jika ada nomor dan list, tolong gunakan format yang sesuai.
#             - Jawaban hanya boleh berdasarkan konteks yang tersedia.
#             - Jika pasal dalam konteks telah diubah, tambahkan keterangan bahwa isi tersebut merupakan hasil amandemen dan sebutkan UU yang mengubahnya.
#             - Jika pasal telah dihapus, sebutkan bahwa pasal tersebut sudah tidak berlaku dan tidak perlu dijelaskan lebih lanjut.
#             - Sebutkan pasal dan ayat secara eksplisit jika tersedia.
#             - Jangan menambahkan interpretasi atau opini di luar kutipan.
#             - Jika tidak ditemukan informasi yang relevan dalam konteks, jawab: "Informasi tidak tersedia dalam konteks."

#             Format konteks:
#             - Diawali dengan nama **Bab** (jika ada) dan **Nomor Pasal**, misalnya `bab ix hubungan kerja - pasal 50`.
#             - Diikuti isi pasal dan ayat, termasuk penanda seperti `(1)`, `(2)`, dst.
#             - Pasal yang diamandemen, ditambahkan, atau dihapus memiliki catatan tambahan dalam teks, misalnya:
#                 - `Pasal 151 (diubah oleh UU No. 6 Tahun 2023)`
#                 - `Pasal 151A (ditambahkan oleh UU No. 6 Tahun 2023)`
#                 - `Pasal 152 (dihapus oleh UU No. 6 Tahun 2023)`

#             Instruksi penting:
#             - Jika pasal merupakan hasil perubahan dari UU lain, beri tahu pengguna bahwa pasal tersebut adalah hasil amandemen dari pasal sebelumnya.
#             - Jika ada pasal baru (misal 151A), nyatakan bahwa pasal tersebut merupakan tambahan dari UU perubahan.
#             - Jika pasal dihapus, cukup nyatakan bahwa pasal tersebut sudah tidak berlaku.

#             <|user|>
#             Berikut adalah konteks dari dokumen hukum yang relevan (kutipan dari UU No. 13 Tahun 2003 dan perubahannya):

#             {context}

#             Pertanyaan:
#             {question}
#             <|assistant|>
#             /no_think
#             """
#         prompt = ChatPromptTemplate.from_template(template)
#         chain = prompt | llm | StrOutputParser()
#         result = chain.invoke({"context": context, "question": query})
#         return {
#             "response": result,
#             "context_chunks": chunks,
#             "metadata": metadata,
#             "tfidf_chunks": context_data["tfidf_chunks"],
#             "embed_chunks": context_data["embed_chunks"]
#         }
#     except Exception as e:
#         return {
#             "response": f"Error saat memproses dengan RAG: {str(e)}",
#             "context_chunks": [],
#             "metadata": []
#         }
        

def get_no_rag_response(query: str):
    llm = OllamaLLM(
            model="qwen3:8b", 
            temperature=0.1, 
            verbose=True)

    try:
        template = """<|begin_of_text|>
        <|system|>
        Kamu asisten hukum yang hanya menjawab berdasarkan peraturan ketenagakerjaan Indonesia khususnya Undang-Undang Nomor 13 Tahun 2003 tentang Ketenagakerjaan beserta perubahannya, termasuk dari UU No. 6 Tahun 2023 (Cipta Kerja). Jawab singkat, sebutkan pasal/ayat dan jelaskan isinya. Jika tidak tahu, jawab: tidak tahu. Jika pasal hasil amandemen, beri tahu.
        <|user|>
        Pertanyaan:
        {question}
        <|assistant|> /no_think"""

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
            return get_rag_response(request.query)
        elif request.mode == "no_rag":
            return get_no_rag_response(request.query)
        else:
            return {
                "rag": get_rag_response(request.query),
                "no_rag": get_no_rag_response(request.query)
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)