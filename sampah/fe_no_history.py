import streamlit as st
import requests

st.set_page_config(page_title="QA UU Ketenagakerjaan", layout="wide")

# Sidebar untuk pemilihan mode
st.sidebar.title("🔧 Pengaturan")
mode = st.sidebar.radio("Pilih Mode Jawaban", options=["rag", "no_rag", "both"], format_func=lambda x: {
    "rag": "Dengan RAG",
    "no_rag": "Tanpa RAG",
    "both": "Bandingkan Keduanya"
}[x])

# Header halaman
st.markdown("<h2 style='text-align: center;'>Tanya Jawab UU No. 13 Tahun 2003 tentang Ketenagakerjaan</h2>", unsafe_allow_html=True)

# Input pertanyaan pengguna
query = st.text_input("Masukkan pertanyaan Anda:")

if st.button("Tanyakan"):
    with st.spinner("🔄 Mengambil jawaban dan konteks..."):
        try:
            response = requests.post("http://127.0.0.1:8000/ask", json={
                "query": query,
                "mode": mode
            })

            if response.status_code == 200:
                data = response.json()

                def tampilkan_hasil(label, jawaban, chunks, tfidf_chunks=None, embed_chunks=None):
                    st.markdown(f"### {label}")
                    st.markdown(jawaban, unsafe_allow_html=True)

                    if chunks:
                        with st.expander("Konteks Final (Hybrid RAG)"):
                            for i, chunk in enumerate(chunks, 1):
                                st.code(chunk, language="text")

                    if tfidf_chunks:
                        with st.expander("Hasil TF-IDF Retriever"):
                            for i, chunk in enumerate(tfidf_chunks, 1):
                                st.code(chunk, language="text")

                    if embed_chunks:
                        with st.expander("Hasil Embedding Retriever"):
                            for i, chunk in enumerate(embed_chunks, 1):
                                st.code(chunk, language="text")

                if mode == "both":
                    col1, col2 = st.columns(2)

                    with col1:
                        tampilkan_hasil(
                            "Jawaban RAG",
                            data["rag"]["response"],
                            data["rag"].get("context_chunks", []),
                            data["rag"].get("tfidf_chunks"),
                            data["rag"].get("embed_chunks")
                        )

                    with col2:
                        tampilkan_hasil(
                            "Jawaban Non-RAG",
                            data["no_rag"]["response"],
                            data["no_rag"].get("context_chunks", [])
                        )

                elif mode == "rag":
                    tampilkan_hasil(
                        "Jawaban RAG",
                        data["response"],
                        data.get("context_chunks", []),
                        data.get("tfidf_chunks"),
                        data.get("embed_chunks")
                    )

                else:  # no_rag
                    tampilkan_hasil(
                        "Jawaban Non-RAG",
                        data["response"],
                        data.get("context_chunks", [])
                    )
            else:
                st.error(f"Terjadi kesalahan: {response.text}")
        except Exception as e:
            st.error(f"Gagal memproses pertanyaan: {str(e)}")
