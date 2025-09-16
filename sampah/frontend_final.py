import streamlit as st
import requests

st.set_page_config(page_title="QA UU Ketenagakerjaan", layout="wide")

st.sidebar.title("Pengaturan")
mode = st.sidebar.radio("Pilih Mode Jawaban", ["rag", "no_rag", "both"], format_func=lambda x: {
    "rag": "Dengan RAG", "no_rag": "Tanpa RAG", "both": "Bandingkan Keduanya"
}[x])

st.markdown("<h2 style='text-align: center;'>Tanya Jawab UU No. 13 Tahun 2003</h2>", unsafe_allow_html=True)

query = st.text_input("Masukkan pertanyaan Anda:")

if st.button("Tanyakan"):
    with st.spinner("🔄 Mengambil jawaban dan konteks..."):
        try:
            response = requests.post("http://127.0.0.1:8000/ask", json={"query": query, "mode": mode})
            if response.status_code == 200:
                data = response.json()

                def tampilkan_hasil(label, jawaban, chunks, tfidf_chunks=None, embed_chunks=None, scores=None, tfidf_scores=None, embed_scores=None):
                    st.markdown(f"### {label}")
                    st.markdown(jawaban, unsafe_allow_html=True)

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

                if mode == "both":
                    col1, col2 = st.columns(2)
                    with col1:
                        tampilkan_hasil("Jawaban RAG", data["rag"]["response"],
                            data["rag"].get("chunks"),
                            data["rag"].get("tfidf_chunks"),
                            data["rag"].get("embed_chunks"),
                            data["rag"].get("scores"),
                            data["rag"].get("tfidf_scores"),
                            data["rag"].get("embed_scores"))
                    with col2:
                        tampilkan_hasil("Jawaban Non-RAG", data["no_rag"]["response"], data["no_rag"].get("chunks"))
                elif mode == "rag":
                    tampilkan_hasil("Jawaban RAG", data["response"],
                        data.get("chunks"),
                        data.get("tfidf_chunks"),
                        data.get("embed_chunks"),
                        data.get("scores"),
                        data.get("tfidf_scores"),
                        data.get("embed_scores"))
                else:
                    tampilkan_hasil("Jawaban Non-RAG", data["response"], data.get("chunks"))
            else:
                st.error(f"Terjadi kesalahan: {response.text}")
        except Exception as e:
            st.error(f"Gagal memproses pertanyaan: {str(e)}")
