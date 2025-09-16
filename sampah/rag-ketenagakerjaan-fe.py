import streamlit as st
import time
import requests

st.set_page_config(page_title="Sistem QA Regulasi Ketenagakerjaan", layout="wide")

# Custom CSS untuk mempercantik tampilan
st.markdown("""
<style>
    /* Style untuk ekspander konteks */
    .stExpander {
        border-left: 3px solid #2c7be5;
        padding-left: 10px;
        margin-top: 10px;
    }
    
    /* Style untuk referensi UU */
    .reference-header {
        background-color: #e6f3ff;
        padding: 8px;
        border-left: 4px solid #2c7be5;
        border-radius: 0 4px 4px 0;
        margin-top: 10px;
        font-weight: bold;
    }
    
    /* Style untuk metadata */
    .metadata-container {
        font-size: 0.8em;
        color: #666;
        margin-bottom: 5px;
        border-bottom: 1px dashed #ddd;
        padding-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Sistem Question Answering Regulasi Ketenagakerjaan Indonesia ⚖️")
st.subheader("UU No. 13 Tahun 2003")

# Sidebar
with st.sidebar:
    st.header("Pengaturan Tampilan")
    show_context = st.toggle("Tampilkan Konteks", value=True)
    show_debug = st.toggle("Tampilkan Debug Info", value=False)
    typing_effect = st.toggle("Efek Mengetik", value=True)
    typing_speed = st.slider("Kecepatan Mengetik", min_value=0.001, max_value=0.05, value=0.01, step=0.001, 
                            help="Semakin kecil nilai, semakin cepat efek mengetik")

# Inisialisasi session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # List of dicts: {"role": ..., "content": ..., "context_chunks": [...], "metadata": [...]}

# Kontainer untuk chat
chat_container = st.container()

# Tampilkan pesan awal jika belum ada percakapan
with chat_container:
    if not st.session_state.messages:
        st.chat_message("assistant").markdown("""
        Halo! 👋 Saya adalah asisten virtual yang dapat membantu Anda dengan informasi tentang regulasi ketenagakerjaan di Indonesia, khususnya UU No. 13 Tahun 2003.

        Anda dapat bertanya tentang:
        - Upah minimum dan sistem pengupahan
        - Hak dan kewajiban pekerja
        - Hubungan kerja dan perjanjian kerja
        - Pemutusan hubungan kerja (PHK)
        - Dan topik ketenagakerjaan lainnya

        Silakan ajukan pertanyaan Anda!
        """)

    # Tampilkan semua pesan (termasuk konteks jika diaktifkan)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if show_context and msg["role"] == "assistant" and "context_chunks" in msg:
                if msg["context_chunks"]:
                    with st.expander("Lihat Konteks UU"):
                        for i, chunk in enumerate(msg["context_chunks"]):
                            # Tampilkan metadata jika tersedia
                            if "metadata" in msg and i < len(msg["metadata"]):
                                meta = msg["metadata"][i]
                                
                                # Informasi referensi
                                bab_nomor = meta.get("bab_nomor", "")
                                bab_judul = meta.get("bab_judul", "")
                                pasal_nomor = meta.get("pasal_nomor", "")
                                reference = meta.get("full_reference", f"{bab_nomor} {bab_judul} - {pasal_nomor}")
                                
                                # Tampilkan referensi dengan styling HTML
                                st.markdown(f'<div class="reference-header">{reference}</div>', unsafe_allow_html=True)
                                
                                # Tampilkan detail tambahan dalam container
                                st.markdown('<div class="metadata-container">', unsafe_allow_html=True)
                                col1, col2, col3 = st.columns([2,1,1])
                                with col1:
                                    st.caption(f"Sumber: {meta.get('source', 'N/A')}")
                                with col2:
                                    st.caption(f"Bagian: {meta.get('chunk', 'N/A')}")
                                with col3:
                                    st.caption(f"Word Count: {meta.get('word_count', '0')}")
                                st.markdown('</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f"**Chunk #{i+1}**")
                            
                            # Tampilkan konten
                            st.code(chunk, language="markdown")
                else:
                    st.info("Tidak ada konteks yang tersedia untuk jawaban ini.")

# Input pertanyaan
prompt = st.chat_input("Tanyakan tentang regulasi ketenagakerjaan...")
if prompt:
    with chat_container:
        # Simpan pertanyaan user
        st.session_state.messages.append({
            "role": "human",
            "content": prompt
        })
        with st.chat_message("human"):
            st.markdown(prompt)

        # Kirim permintaan ke backend
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🔍 Mencari informasi...")

            try:
                url = "http://127.0.0.1:8000/ask"
                history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])

                response = requests.post(url, json={
                    "query": prompt,
                    "history": history_text
                })

                if response.status_code == 200:
                    data = response.json()

                    # Debug respons backend
                    if show_debug:
                        with st.expander("🔎 Debug Respons Backend"):
                            st.json(data)

                    answer = data.get("response", "Maaf, tidak ada jawaban.")
                    context_chunks = data.get("context_chunks", [])
                    metadata = data.get("metadata", [])

                    # Efek mengetik jika diaktifkan
                    if typing_effect:
                        typed = ""
                        for word in answer.split():
                            typed += word + " "
                            message_placeholder.markdown(typed)
                            time.sleep(typing_speed)
                    
                    # Gantikan placeholder dengan hasil akhir
                    message_placeholder.markdown(answer)

                    # Simpan jawaban + konteks
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "context_chunks": context_chunks,
                        "metadata": metadata
                    })
                else:
                    message_placeholder.error("❌ Gagal menghubungi server. Status: " + str(response.status_code))

            except Exception as e:
                message_placeholder.error(f"❌ Error: {e}")