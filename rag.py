import re
import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import TFIDFRetriever
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load environment and initialize
_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
DEFAULT_PERSIST_DIR = "./chroma_db_openai_gabungan_pasal_cosine_1"

# Build hybrid retriever
def build_hybrid_retriever(embedding_model, tfidf_k=2, embed_k=2, weights=[0.7, 0.3], persist_directory=DEFAULT_PERSIST_DIR):
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    raw_docs = db.get(include=["documents", "metadatas"])
    documents = [Document(page_content=doc, metadata=meta) for doc, meta in zip(raw_docs["documents"], raw_docs["metadatas"])]

    tfidf_retriever = TFIDFRetriever.from_documents(documents)
    tfidf_retriever.k = tfidf_k

    embedding_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": embed_k})
    hybrid_retriever = EnsembleRetriever(retrievers=[embedding_retriever, tfidf_retriever], weights=weights, k=2)

    return hybrid_retriever

# Calculate scores
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

# Get context
def get_retriever_context(input_query):
    retriever = build_hybrid_retriever(embedding_model=embeddings)
    tfidf_results = retriever.retrievers[1].invoke(input_query)
    embed_results = retriever.retrievers[0].invoke(input_query)
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
# Display results in Streamlit
def tampilkan_hasil(chunks, tfidf_chunks=None, embed_chunks=None, scores=None, tfidf_scores=None, embed_scores=None):
    if chunks:
        with st.expander("Konteks Final (Hybrid RAG)"):
            for chunk in chunks:
                hybrid = scores.get(chunk, 0) if scores else None
                tfidf = tfidf_scores.get(chunk, 0) if tfidf_scores else None
                embed = embed_scores.get(chunk, 0) if embed_scores else None

                score_text = f"\n[Skor RFF: {hybrid:.4f}]"
                st.code(chunk + score_text, language="text")

    if tfidf_chunks:
        with st.expander("Hasil TF-IDF Retriever"):
            for chunk in tfidf_chunks:
                tfidf = tfidf_scores.get(chunk, 0) if tfidf_scores else None
                st.code(f"{chunk}\n[TF-IDF: {tfidf:.4f}]", language="text")

    if embed_chunks:
        with st.expander("Hasil Embedding Retriever"):
            for chunk in embed_chunks:
                embed = embed_scores.get(chunk, 0) if embed_scores else None
                st.code(f"{chunk}\n[Embedding: {embed:.4f}]", language="text")


# RAG and non-RAG response generators
def stream_rag_response(query):
    llm = OllamaLLM(model="qwen3:8b", temperature=0)
    context_data = get_retriever_context(query)
    context = context_data["context"]

    template = """<|im_start|>system
            Kamu adalah asisten hukum spesialis ketenagakerjaan di Indonesia. Tugasmu adalah menjawab pertanyaan pengguna secara singkat dan hanya berdasarkan kutipan resmi dari Undang-Undang yang tersedia dalam konteks, yaitu UU No. 13 Tahun 2003 tentang Ketenagakerjaan dan perubahannya, termasuk UU No. 6 Tahun 2023 (Cipta Kerja).

            Aturan ketat dalam menjawab:
            - Jawaban diawali dengan penjelasan kemudian diikuti dengan kutipan pasal/ayat yang relevan berdasarkan konteks yang diberikan.
            - Jika ada nomor dan list, tolong gunakan format yang sesuai.
            - Jawaban hanya boleh berdasarkan konteks yang tersedia.
            - Jika pasal dalam konteks telah diubah, tambahkan keterangan bahwa isi tersebut merupakan hasil amandemen dan sebutkan UU yang mengubahnya.
            - Jika pasal telah dihapus, sebutkan bahwa pasal tersebut sudah tidak berlaku dan tidak perlu dijelaskan lebih lanjut.
            - Sebutkan pasal dan ayat secara eksplisit jika tersedia.
            - Jangan menambahkan interpretasi atau opini di luar kutipan.
            - Jika tidak ditemukan informasi yang relevan dalam konteks, jawab: "Maaf saya tidak bisa menjawab dikarenakan informasi tidak tersedia dalam konteks."

            Format konteks:
            - Dimulai dengan nama Bab, contoh: bab ix Hubungan Kerja.
            - Diikuti isi pasal dan ayat: (1), (2), dst
            - Ayat didefinisikan dalam tanda kurung misalnya ayat 1 berati (1)
            - Point-poin dalam pasal/ayat ditandai dengan tanda titik dua (:) dan diakhiri dengan titik koma (;)
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
            (Jawab hanya jika ditemukan kutipan pasal yang relevan dalam konteks)
            {question}
            <|im_end|>
            <|im_start|>assistant
            /no_think
            """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.stream({"context": context, "question": query}), context_data

def get_no_rag_response(query):
    llm = OllamaLLM(model="qwen3:8b", temperature=0.1)
    template = """
        <|im_start|>system
        Kamu asisten hukum yang hanya menjawab berdasarkan peraturan ketenagakerjaan Indonesia khususnya Undang-Undang Nomor 13 Tahun 2003 tentang Ketenagakerjaan beserta perubahannya, termasuk dari UU No. 6 Tahun 2023 (Cipta Kerja). Jawab singkat, sebutkan pasal/ayat dan jelaskan isinya. Jika tidak tahu, jawab: tidak tahu. Jika pasal hasil amandemen, beri tahu.
        <|im_end|>
        <|im_start|>user
        Pertanyaan:
        {question}
        <|im_end|>
        <|im_start|>assistant
        /no_think
        """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.stream({"question": query})
    # return chain.invoke({"question": query})

# Streamlit UI
st.set_page_config(page_title="QA UU Ketenagakerjaan", layout="wide")
st.sidebar.title("Pengaturan")
mode = st.sidebar.radio("Mode Jawaban", ["rag", "no_rag", "both"], format_func=lambda x: {"rag": "Dengan RAG", "no_rag": "Tanpa RAG", "both": "Keduanya"}[x])
st.markdown("<h2 style='text-align: center;'>Tanya Jawab UU Ketenagakerjaan</h2>", unsafe_allow_html=True)

query = st.text_input("Masukkan pertanyaan Anda:")
if st.button("Tanyakan") and query:
    with st.spinner("🔄 Mengambil jawaban dan konteks..."):
        if mode in ["rag", "both"]:
            st.markdown("### Jawaban dengan RAG")
            rag_stream, context_data = stream_rag_response(query)
            placeholder = st.empty()
            output = ""
            for token in rag_stream:
                output += token
                clean_output =  re.sub(r"</?think>", "", output)
                placeholder.markdown(clean_output)

            tampilkan_hasil(
                chunks=context_data.get("chunks"),
                tfidf_chunks=context_data.get("tfidf_chunks"),
                embed_chunks=context_data.get("embed_chunks"),
                scores=context_data.get("scores"),
                tfidf_scores=context_data.get("tfidf_scores"),
                embed_scores=context_data.get("embed_scores")
            )

        if mode in ["no_rag", "both"]:
            st.markdown("### Jawaban tanpa RAG")
            no_rag_stream = get_no_rag_response(query)
            placeholder = st.empty()
            output = ""
            for token in no_rag_stream:
                output += token
                clean_output =  re.sub(r"</?think>", "", output)
                placeholder.markdown(clean_output)