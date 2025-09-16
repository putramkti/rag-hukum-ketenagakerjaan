import streamlit as st
import time
import requests
from typing import Optional, Dict, Any, List

st.set_page_config(
    page_title="Sistem QA Regulasi Ketenagakerjaan - Perbandingan RAG", 
    layout="wide"
)

# Custom CSS untuk mempercantik tampilan
st.markdown("""
<style>
    /* Base styling for the page */
    .main {
        # background-color: #f5f7fa;
    }
    
    /* Header style */
    .header-container {
        background-color: #2c7be5;
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    /* Response container */
    .response-container {
        border: 1px solid #e0e5ec;
        border-radius: 8px;
        padding: 1rem;
        background-color: white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        height: 100%;
    }
    
    /* RAG response highlight */
    .rag-container {
        border-left: 4px solid #2c7be5;
    }
    
    /* Non-RAG response highlight */
    .no-rag-container {
        border-left: 4px solid #e63946;
    }
    
    /* Style untuk ekspander konteks */
    .stExpander {
        border-left: 3px solid #2c7be5;
        padding-left: 10px;
        margin-top: 10px;
    }
    
    /* Style untuk referensi UU */
    .reference-header {
        # background-color: #e6f3ff;
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
        # border-bottom: 1px dashed #ddd;
        padding-bottom: 5px;
    }
    
    /* Response header */
    .response-header {
        font-weight: bold;
        padding: 5px 10px;
        border-radius: 4px;
        margin-bottom: 10px;
        text-align: center;
    }
    
    .rag-header {
        # background-color: #e6f3ff;
        color: #0a59c0;
    }
    
    .no-rag-header {
        # background-color: #ffebee;
        color: #c62828;
    }
    
    /* Loading indicator */
    .loading {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px;
        font-style: italic;
        color: #666;
    }
    
    /* Response content */
    .response-content {
        padding: 10px;
        border-radius: 4px;
        # background-color: #f9f9f9;
    }
    
    /* History message container */
    .message-human {
        # background-color: #f0f7ff;
        border-radius: 15px 15px 0 15px;
        padding: 10px 15px;
        margin: 5px 0;
        text-align: right;
        max-width: 80%;
        margin-left: auto;
    }
    
    .message-assistant-rag {
        # background-color: #e6f3ff;
        border-left: 3px solid #2c7be5;
        border-radius: 15px 15px 15px 0;
        padding: 10px 15px;
        margin: 5px 0;
        max-width: 80%;
    }
    
    .message-assistant-no-rag {
        # background-color: #fff0f0;
        border-left: 3px solid #e63946;
        border-radius: 15px 15px 15px 0;
        padding: 10px 15px;
        margin: 5px 0;
        max-width: 80%;
    }
    
    /* Compare button */
    .compare-button {
        background-color: #2c7be5;
        color: white;
        border-radius: 4px;
        padding: 10px 15px;
        font-weight: bold;
        text-align: center;
        margin: 15px 0;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# Header dengan HTML styling
st.markdown("""
<div class="header-container">
    <h1>Sistem QA Regulasi Ketenagakerjaan Indonesia ⚖️</h1>
    <h3>Perbandingan Respon: Dengan RAG vs Tanpa RAG</h3>
</div>
""", unsafe_allow_html=True)

show_context = True
show_debug = False
typing_effect = True
typing_speed = 0.01
api_endpoint = "http://127.0.0.1:8000/ask"

# Sidebar
# with st.sidebar:
#     st.header("Pengaturan Tampilan")
#     show_context = st.toggle("Tampilkan Konteks", value=True)
#     show_debug = st.toggle("Tampilkan Debug Info", value=False)
#     typing_effect = st.toggle("Efek Mengetik", value=True)
#     typing_speed = st.slider("Kecepatan Mengetik", min_value=0.001, max_value=0.05, value=0.01, step=0.001, 
#                             help="Semakin kecil nilai, semakin cepat efek mengetik")
    
#     st.markdown("---")
#     st.header("API Endpoint")
#     api_endpoint = st.text_input("Endpoint API", value="http://127.0.0.1:8000/ask")

# Inisialisasi session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # List of dicts: {"query": ..., "rag_response": ..., "no_rag_response": ..., "context": ...}

# Tampilkan pesan awal jika belum ada percakapan
if not st.session_state.messages:
    st.info("""
    👋 Selamat datang di sistem perbandingan jawaban regulasi ketenagakerjaan!
    
    Silakan ajukan pertanyaan tentang regulasi ketenagakerjaan di Indonesia!
    """)

# Fungsi untuk melakukan request ke backend terpadu
def get_unified_response(query: str, history: str, endpoint: str) -> Dict[str, Any]:
    try:
        response = requests.post(
            endpoint,
            json={"query": query, "history": history, "mode": "both"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Ekstrak data sesuai dengan format respons yang terpadu
            if "rag" in result and "no_rag" in result:
                # Format respons lengkap untuk keduanya
                return {
                    "rag_response": result["rag"]["response"],
                    "no_rag_response": result["no_rag"]["response"],
                    "context_chunks": result["rag"]["context_chunks"],
                    "metadata": result["rag"]["metadata"]
                }
            else:
                # Fallback jika struktur tidak sesuai
                return {
                    "rag_response": "Error: Format respons tidak sesuai",
                    "no_rag_response": "Error: Format respons tidak sesuai",
                    "context_chunks": [],
                    "metadata": []
                }
        else:
            return {
                "rag_response": f"Error: Status code {response.status_code}",
                "no_rag_response": f"Error: Status code {response.status_code}",
                "context_chunks": [],
                "metadata": []
            }
    except Exception as e:
        return {
            "rag_response": f"Error: {str(e)}",
            "no_rag_response": f"Error: {str(e)}",
            "context_chunks": [],
            "metadata": []
        }

# Fungsi untuk tampilkan respon dengan efek typing
def display_response_with_typing(container, response_text: str, typing_effect: bool, typing_speed: float):
    if typing_effect:
        full_response = ""
        for word in response_text.split():
            full_response += word + " "
            container.markdown(full_response)
            time.sleep(typing_speed)
    container.markdown(response_text)

# Container untuk history percakapan
history_container = st.container()

# Tampilkan history percakapan
with history_container:
    for msg in st.session_state.messages:
        # Tampilkan pertanyaan user
        st.markdown(f'<div class="message-human">{msg["query"]}</div>', unsafe_allow_html=True)
        
        # Tampilkan respon dengan RAG dan non-RAG
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f'<div class="message-assistant-rag">{msg["rag_response"]}</div>', unsafe_allow_html=True)
            
            # Tampilkan konteks jika diaktifkan
            if show_context and "context_chunks" in msg and msg["context_chunks"]:
                with st.expander("Lihat Konteks UU (RAG)"):
                    for i, chunk in enumerate(msg["context_chunks"]):
                        # Tampilkan metadata jika tersedia
                        if "metadata" in msg and i < len(msg["metadata"]):
                            meta = msg["metadata"][i]
                            
                            # Informasi referensi
                            reference = meta.get("full_reference", f"Referensi #{i+1}")
                            
                            # Tampilkan referensi dengan styling
                            st.markdown(f'<div class="reference-header">{reference}</div>', unsafe_allow_html=True)
                            
                            # Tampilkan detail tambahan
                            st.markdown('<div class="metadata-container">', unsafe_allow_html=True)
                            col1a, col2a, col3a = st.columns([2,1,1])
                            with col1a:
                                st.caption(f"Sumber: {meta.get('source', 'N/A')}")
                            with col2a:
                                st.caption(f"Bagian: {meta.get('chunk', 'N/A')}")
                            with col3a:
                                st.caption(f"Word Count: {meta.get('word_count', '0')}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Tampilkan konten
                        # Tampilkan konten dengan word wrap
                        st.markdown(f"""
                        <div style="white-space: pre-wrap; font-family: monospace; background-color: #f8f9fa; padding: 10px; border-radius: 4px; border: 1px solid #ddd;">
                        {chunk}
                        </div>
                        """, unsafe_allow_html=True)

        
        with col2:
            st.markdown(f'<div class="message-assistant-no-rag">{msg["no_rag_response"]}</div>', unsafe_allow_html=True)

# Input pertanyaan
prompt = st.chat_input("Tanyakan tentang regulasi ketenagakerjaan...")

if prompt:
    # Simpan pertanyaan user
    st.markdown(f'<div class="message-human">{prompt}</div>', unsafe_allow_html=True)
    
    # Kontainer untuk respons
    col1, col2 = st.columns(2)
    
    # Generate history text untuk konteks
    history_text = "\n".join([
        f"human: {msg['query']}\nassistant_rag: {msg['rag_response']}\nassistant_no_rag: {msg['no_rag_response']}"
        for msg in st.session_state.messages
    ])
    
    # Placeholder untuk respons
    with col1:
        st.markdown('<div class="response-header rag-header">Respon dengan RAG</div>', unsafe_allow_html=True)
        rag_placeholder = st.empty()
        rag_placeholder.markdown('<div class="loading">🔍 Mencari informasi dengan RAG...</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="response-header no-rag-header">Respon tanpa RAG</div>', unsafe_allow_html=True)
        no_rag_placeholder = st.empty()
        no_rag_placeholder.markdown('<div class="loading">💭 Menghasilkan respons tanpa RAG...</div>', unsafe_allow_html=True)
    
    # Get responses from unified backend
    result = get_unified_response(prompt, history_text, api_endpoint)
    
    # Display debug info if enabled
    if show_debug:
        debug_container = st.expander("🔎 Debug Info")
        with debug_container:
            st.json(result)
    
    # Extract responses
    rag_response = result.get("rag_response", "Maaf, tidak ada jawaban dari sistem RAG.")
    no_rag_response = result.get("no_rag_response", "Maaf, tidak ada jawaban dari sistem non-RAG.")
    context_chunks = result.get("context_chunks", [])
    metadata = result.get("metadata", [])
    
    # Display responses with typing effect
    with col1:
        display_response_with_typing(rag_placeholder, rag_response, typing_effect, typing_speed)
        
        # Show context if enabled
        if show_context and context_chunks:
            with st.expander("Lihat Konteks UU (RAG)"):
                for i, chunk in enumerate(context_chunks):
                    # Display metadata if available
                    if i < len(metadata):
                        meta = metadata[i]
                        
                        # Reference information
                        reference = meta.get("full_reference", f"Referensi #{i+1}")
                        
                        # Display reference with styling
                        st.markdown(f'<div class="reference-header">{reference}</div>', unsafe_allow_html=True)
                        
                        # Display additional details
                        st.markdown('<div class="metadata-container">', unsafe_allow_html=True)
                        col1a, col2a, col3a = st.columns([2,1,1])
                        with col1a:
                            st.caption(f"Sumber: {meta.get('source', 'N/A')}")
                        with col2a:
                            st.caption(f"Bagian: {meta.get('chunk', 'N/A')}")
                        with col3a:
                            st.caption(f"Word Count: {meta.get('word_count', '0')}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display content
                    st.code(chunk, language="markdown")
    
    with col2:
        display_response_with_typing(no_rag_placeholder, no_rag_response, typing_effect, typing_speed)
    
    # Save to session state
    st.session_state.messages.append({
        "query": prompt,
        "rag_response": rag_response,
        "no_rag_response": no_rag_response,
        "context_chunks": context_chunks,
        "metadata": metadata
    })

# Tambahkan tombol reset di sidebar
# with st.sidebar:
#     if st.button("Reset Percakapan", type="primary"):
#         st.session_state.messages = []
#         st.experimental_rerun()