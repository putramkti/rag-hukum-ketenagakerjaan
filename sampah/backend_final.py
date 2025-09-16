import os
from dotenv import load_dotenv, find_dotenv
import openai
from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import TFIDFRetriever
from langchain_core.documents import Document
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
DEFAULT_PERSIST_DIR = "./chroma_db_openai_gabungan_pasal_cosine"

app = FastAPI(title="Ketenagakerjaan API", version="0.1")

class QueryRequest(BaseModel):
    query: str
    mode: Optional[str] = "both"

def build_hybrid_retriever(embedding_model, tfidf_k=2, embed_k=2, weights=[0.7, 0.3], persist_directory=DEFAULT_PERSIST_DIR):
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    raw_docs = db.get(include=["documents", "metadatas"])
    documents = [Document(page_content=doc, metadata=meta) for doc, meta in zip(raw_docs["documents"], raw_docs["metadatas"])]

    tfidf_retriever = TFIDFRetriever.from_documents(documents)
    tfidf_retriever.k = tfidf_k

    embedding_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": embed_k})

    hybrid_retriever = EnsembleRetriever(retrievers=[embedding_retriever, tfidf_retriever], weights=weights, k=2)
    return hybrid_retriever

def compute_similarity_scores(query, tfidf_results, embed_results, embedding_model):
    tfidf_vectorizer = TfidfVectorizer()
    docs_tfidf = tfidf_vectorizer.fit_transform([query] + [doc.page_content for doc in tfidf_results])
    tfidf_similarities = cosine_similarity(docs_tfidf[0:1], docs_tfidf[1:]).flatten()
    tfidf_scores = {doc.page_content: float(score) for doc, score in zip(tfidf_results, tfidf_similarities)}

    embed_query = embedding_model.embed_query(query)
    embed_docs = [embedding_model.embed_query(doc.page_content) for doc in embed_results]
    embed_similarities = cosine_similarity([embed_query], embed_docs).flatten()
    embed_scores = {doc.page_content: float(score) for doc, score in zip(embed_results, embed_similarities)}

    return tfidf_scores, embed_scores

def get_retriever_context(input_query):
    try:
        retriever = build_hybrid_retriever(embedding_model=embeddings)
        tfidf_results = retriever.retrievers[1].invoke(input_query)
        embed_results = retriever.retrievers[0].invoke(input_query)
        print(embed_results)
        final_results = retriever.invoke(input_query)

        tfidf_scores, embed_scores = compute_similarity_scores(input_query, tfidf_results, embed_results, embeddings)

        def compute_rrf_scores(docs):
            return {doc.page_content: 1 / (rank + 1 + 60) for rank, doc in enumerate(docs)}

        rrf_embed = compute_rrf_scores(embed_results)
        rrf_tfidf = compute_rrf_scores(tfidf_results)
        combined_scores = {
            doc.page_content: 0.7 * rrf_embed.get(doc.page_content, 0) + 0.3 * rrf_tfidf.get(doc.page_content, 0)
            for doc in final_results
        }

        return {
            "context": "\n\n".join([doc.page_content for doc in final_results]),
            "chunks": [doc.page_content for doc in final_results],
            "metadata": [doc.metadata for doc in final_results],
            "tfidf_chunks": [doc.page_content for doc in tfidf_results],
            "embed_chunks": [doc.page_content for doc in embed_results],
            "scores": combined_scores,
            "tfidf_scores": tfidf_scores,
            "embed_scores": embed_scores,
        }
    except Exception as e:
        return {
            "context": f"Error: {str(e)}",
            "chunks": [], "metadata": [],
            "tfidf_chunks": [], "embed_chunks": [],
            "scores": {}, "tfidf_scores": {}, "embed_scores": {}
        }

def get_rag_response(query: str):
    llm = OllamaLLM(model="qwen3:8b", temperature=0, verbose=True)
    context_data = get_retriever_context(query)
    context = context_data["context"]

    template = """<|im_start|>system
            Kamu adalah asisten hukum spesialis ketenagakerjaan di Indonesia. Tugasmu adalah menjawab pertanyaan pengguna secara singkat dan hanya berdasarkan kutipan resmi dari Undang-Undang yang tersedia dalam konteks, yaitu UU No. 13 Tahun 2003 tentang Ketenagakerjaan dan perubahannya, termasuk UU No. 6 Tahun 2023 (Cipta Kerja).

            Aturan ketat dalam menjawab:
            - Jawab secara singkat dan langsung pada kesimpulan berdasarkan konteks yang diberikan.
            - Jawaban hanya berdasarkan isi pasal dan ayat yang tersedia di konteks.
            - Setiap jawaban harus menyebutkan Pasal dan Ayat secara eksplisit, dan menjelaskan isinya sesuai kutipan.
            - Jika pasal telah diubah, tuliskan isi pasal sesuai versi perubahan dan beri keterangan bahwa itu hasil amandemen, contohnya:
            "Berdasarkan Pasal 151 ayat (1) (diubah oleh UU No. 6 Tahun 2023), …"
            - Jika pasal merupakan tambahan, sebutkan bahwa itu pasal baru dan jelaskan isinya.
            - Jika pasal telah dihapus, JANGAN menjelaskan isinya. Cukup jawab:
            "Pasal X telah dihapus dan tidak lagi berlaku berdasarkan UU No. 6 Tahun 2023."
            - Jangan menambahkan interpretasi, opini pribadi, atau menjawab berdasarkan asumsi.
            - Jika tidak ada informasi relevan di konteks, jawab: "Informasi tidak tersedia dalam konteks."
            - Jika terdapat beberapa pasal yang relevan, urutkan jawaban berdasarkan logika hukum: dimulai dari pasal utama, lalu ke pengecualian, lalu tambahan atau khusus.
            - Gunakan format markdown jika isi pasal/ayat mengandung banyak poin agar mudah dibaca.

            Format konteks:
            - Nama Bab
            - Diikuti isi pasal dalam satu baris, termasuk ayat dan poin jika ada.
            - Perubahan ditandai seperti:
                - Pasal 77 (diubah oleh UU No. 6 Tahun 2023)
                - Pasal 78 (ditambahkan oleh UU No. 6 Tahun 2023)
                - Pasal 152 (dihapus oleh UU No. 6 Tahun 2023)
            
            Tujuan kamu adalah memberikan jawaban hukum yang akurat dan mudah dibaca dengan struktur yang rapi.
            <|im_end|>
            <|im_start|>user
            Berikut adalah konteks dari dokumen hukum yang relevan (kutipan dari UU No. 13 Tahun 2003 dan perubahannya):

            {context}

            Pertanyaan:
            (Jawab hanya jika ditemukan kutipan pasal yang relevan dalam konteks)
            {question}
            <|im_end|>
            <|im_start|>assistant
            /no_think
            """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    # for token in chain.stream({"context": context, "question": query}):
    #     print(token, end="", flush=True)

    # result = chain.stream({"context": context, "question": query})
    # print(f"Result: {result}")
    return {
        "response": [print(token, end="", flush=True) for token in chain.stream({"context": context, "question": query})],

        **context_data
    }

def get_no_rag_response(query: str):
    llm = OllamaLLM(
            model="qwen3:8b", 
            temperature=0.1, 
            verbose=True)
    template = """
        <|system|>
        Kamu asisten hukum yang hanya menjawab berdasarkan peraturan ketenagakerjaan Indonesia khususnya Undang-Undang Nomor 13 Tahun 2003 tentang Ketenagakerjaan beserta perubahannya, termasuk dari UU No. 6 Tahun 2023 (Cipta Kerja). Jawab singkat, sebutkan pasal/ayat dan jelaskan isinya. Jika tidak tahu, jawab: tidak tahu. Jika pasal hasil amandemen, beri tahu.
        <|user|>
        Pertanyaan:
        {question}
        <|assistant|> /no_think"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": query})
    return {"response": result, "context_chunks": []}

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
