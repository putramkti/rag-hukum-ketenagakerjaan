import re
import os
import string
import streamlit as st
import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import openai
import tiktoken
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

import tempfile
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment and initialize
_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
DEFAULT_PERSIST_DIR = "./chroma_db_openai_ketenagakerjaan"

# Initialize tokenizer for preprocessing
tokenizer = tiktoken.get_encoding("cl100k_base")

def num_tokens(text: str) -> int:
    """Count tokens in text"""
    return len(tokenizer.encode(text))

def preprocess_text_for_indexing(text):
    """
    Preprocessing function similar to the original indexing code
    """
    # Potong teks mulai dari BAB I jika ada
    match = re.search(r'\bBAB\s+I\b', text, flags=re.IGNORECASE)
    if match:
        text = text[match.start():]

    # Hapus header tidak perlu
    text = re.sub(r'PRESIDEN\s+REPUBLIK\s+INDONESIA', '', text, flags=re.IGNORECASE)
    text = re.sub(r'-\s*\d+\s*-', '', text)

    # Hapus semua baris yang mengandung '...' atau karakter elipsis Unicode '…'
    text = re.sub(r'^.*(\.{3}|…).*$', '', text, flags=re.MULTILINE)

    # Hapus baris kosong sisa
    text = re.sub(r'^\s*\n', '', text, flags=re.MULTILINE)

    # Normalisasi spasi jadi satu spasi
    text = re.sub(r'\s+', ' ', text)

    # Hapus baris seperti 'Pasal x Cukup jelas'
    text = re.sub(r'\bPasal\s+\d+\s+Cukup jelas\b', '', text, flags=re.IGNORECASE)
    
    # Tambahkan newline sebelum BAB (angka romawi)
    text = re.sub(r'(?<!\n)(BAB\s+[IVXLCDM]+)', r'\n\1', text, flags=re.IGNORECASE)

    # Tambahkan newline sebelum 'Pasal x' yang berdiri sendiri
    text = re.sub(
        r'(?<!\S)(Pasal\s+\d+)\b(?!\s*(ayat\b|,|dan\b))',
        r'\n\1',
        text,
        flags=re.IGNORECASE
    )
    
    # Hilangkan newline pada 'dalam\nPasal x' dan 'dan\nPasal x'
    text = re.sub(r'(dalam|dan)\s*\n\s*(Pasal\s+\d+)', r'\1 \2', text, flags=re.IGNORECASE)

    # Bersihkan spasi di awal dan akhir
    text = text.strip()

    return text

# ========= FUNGSI CHUNKING UNTUK DOKUMEN BARU =========
def chunk_uu_by_token_for_new_doc(
    text: str, min_token: int = 512, max_token: int = 1024, overlap: int = 50, doc_name: str = "Dokumen Baru"
):
    """
    Chunking function identical to the original indexing code
    """
    # Inisialisasi text splitter berbasis token encoder dengan tokenizer cl100k_base (OpenAI).
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=max_token,
        chunk_overlap=overlap,
        separators=[". ", " ", ""] # Urutan preferensi pemisah chunk
    )

    # Temukan semua BAB
    bab_pattern = re.compile(r'^\s*(BAB\s+([IVXLCDM]+)\s+([A-Z\s,/.&-]+))\s*$', re.MULTILINE)
    # Cari semua kecocokan BAB dalam teks
    bab_matches = list(bab_pattern.finditer(text))

    # Inisialisasi daftar list untuk menyimpan dokumen
    documents = []
    if (bab_matches):
        # Iterasi setiap BAB yang ditemukan
        for i, match in enumerate(bab_matches):
            # Posisi akhir BAB saat ini sebagai awal isi
            start = match.end()
            # Sampai awal BAB berikutnya / akhir teks
            end = bab_matches[i + 1].start() if i + 1 < len(bab_matches) else len(text)

            bab_nomor = f"BAB {match.group(2)}"
            bab_judul = match.group(3).strip()
            bab_content = text[start:end].strip()

            # Split isi BAB menjadi bagian per Pasal
            pasal_pattern = re.compile(r'(?:^|\n)(?=Pasal\s+\d+\b)')
            pasal_texts = pasal_pattern.split(bab_content)

            buffer_text = ""
            buffer_pasal = []

            # Iterasi setiap Pasal dalam BAB
            for j, pasal_text in enumerate(pasal_texts):
                pasal_text = pasal_text.strip()
                if not pasal_text:
                    continue

                # Cari dan ambil nomor pasal dari teks
                pasal_match = re.search(r'Pasal\s+(\d+)', pasal_text)
                pasal_nomor = f"Pasal {pasal_match.group(1)}" if pasal_match else f"Pasal-{j+1}"
                
                # Tambahkan teks pasal ke buffer
                buffer_text += "\n" + pasal_text
                buffer_pasal.append(pasal_nomor)
                
                # Hitung token dalam buffer
                token_count = num_tokens(buffer_text)

                # Jika buffer cukup panjang atau sudah di akhir, simpan chunk
                if token_count >= min_token or j == len(pasal_texts) - 1:
                    page_text = f"{doc_name}, {bab_nomor} {bab_judul} :\n{buffer_text.strip()}".lower()
                    
                    # Jika jumlah token melebihi batas maksimum, pisahkan menjadi beberapa chunk
                    if num_tokens(page_text) > max_token:
                        chunks = text_splitter.create_documents([page_text])
                        for k, chunk in enumerate(chunks):
                            # Sisipkan kembali informasi BAB & Pasal di awal isi
                            if k != 0:
                                new_content = f"{doc_name}, {bab_nomor} {bab_judul} :\n {pasal_nomor} {chunk.page_content.strip()}".lower()
                            else:
                                new_content = chunk.page_content.strip()
                            chunk.page_content = new_content
                            # Tambahkan metadata untuk BAB, Pasal, dan jumlah token
                            chunk.metadata = {
                                "bab_nomor": bab_nomor,
                                "bab_judul": bab_judul,
                                "pasal_nomor": ", ".join(buffer_pasal),
                                "jumlah_token": num_tokens(chunk.page_content)
                            }
                            documents.append(chunk)
                    else:
                        # Jika panjang masih di bawah max_token, langsung simpan sebagai satu Document
                        doc = Document(
                            page_content=page_text.strip(),
                            metadata={
                                "bab_nomor": bab_nomor,
                                "bab_judul": bab_judul,
                                "pasal_nomor": ", ".join(buffer_pasal),
                                "jumlah_token": num_tokens(page_text)
                            }
                        )               
                        documents.append(doc)
                    # Reset buffer untuk pasal berikutnya
                    buffer_text = ""
                    buffer_pasal = []

    else:
        pasal_pattern = re.compile(r'(?:^|\n)(?=Pasal\s+\d+\b)')
        pasal_texts = pasal_pattern.split(text)

        buffer_text = ""
        buffer_pasal = []

        for i, pasal_text in enumerate(pasal_texts):
            pasal_text = pasal_text.strip()
            if not pasal_text:
                continue

            pasal_match = re.search(r'Pasal\s+(\d+)', pasal_text)
            pasal_nomor = f"Pasal {pasal_match.group(1)}" if pasal_match else f"Pasal-{i+1}"

            buffer_text += "\n" + pasal_text
            buffer_pasal.append(pasal_nomor)

            token_count = num_tokens(buffer_text)

            if token_count >= min_token or i == len(pasal_texts) - 1:
                page_text = f"{doc_name}:\n{buffer_text.strip()}".lower()
                if num_tokens(page_text) > max_token:
                    chunks = text_splitter.create_documents([page_text])
                    for k, chunk in enumerate(chunks):
                        chunk.page_content = f"{doc_name}:\n{chunk.page_content.strip()}"
                        chunk.metadata = {
                            "pasal_nomor": ", ".join(buffer_pasal),
                            "jumlah_token": num_tokens(chunk.page_content)
                        }
                        documents.append(chunk)
                else:
                    doc = Document(
                        page_content=page_text.strip(),
                        metadata={
                            "pasal_nomor": ", ".join(buffer_pasal),
                            "jumlah_token": num_tokens(page_text)
                        }
                    )
                    documents.append(doc)
                buffer_text = ""
                buffer_pasal = []
    return documents

# ========= FUNGSI CRUD UNTUK DOKUMEN HUKUM =========
def process_uploaded_file(uploaded_file):
    """Memproses file yang diupload dan mengekstrak teks"""
    try:
        if uploaded_file.type == "application/pdf":
            # Simpan file sementara
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name
            
            # Load dan ekstrak teks
            loader = PyPDFLoader(temp_path)
            pages = loader.load()
            
            # Gabungkan semua halaman menjadi satu teks
            full_text = ""
            for page in pages:
                full_text += page.page_content + "\n"
            
            # Hapus file sementara
            os.unlink(temp_path)
            
            return full_text
        else:
            # Untuk file teks
            return str(uploaded_file.read(), "utf-8")
    except Exception as e:
        st.error(f"Error memproses file: {str(e)}")
        return None

# Perbaiki fungsi add_documents_to_chroma (nama fungsi yang benar)
def add_documents_to_chroma(documents, embedding_model, persist_dir=DEFAULT_PERSIST_DIR):
    """Menambahkan dokumen ke Chroma DB dan mengembalikan ID yang ditambahkan"""
    try:
        # Load existing database
        db = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
        
        # Generate unique IDs untuk setiap dokumen
        doc_ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        # Add documents dengan IDs
        db.add_documents(documents, ids=doc_ids)
        
        return True, doc_ids
    except Exception as e:
        st.error(f"Error menambahkan dokumen: {str(e)}")
        return False, []

# Perbaiki fungsi delete_documents_from_chroma
def delete_documents_from_chroma(doc_ids, persist_dir=DEFAULT_PERSIST_DIR):
    """Menghapus dokumen berdasarkan ID dari Chroma DB"""
    try:
        # Load existing database
        db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        
        # Hapus dokumen berdasarkan ID
        db.delete(ids=doc_ids)
        
        return True
    except Exception as e:
        st.error(f"Error menghapus dokumen: {str(e)}")
        return False

# Perbaiki fungsi view_all_documents untuk mengembalikan format yang benar
def view_all_documents(persist_dir=DEFAULT_PERSIST_DIR):
    """Melihat semua dokumen dalam Chroma DB"""
    try:
        db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        
        # Get all documents
        results = db.get(include=["documents", "metadatas", "embeddings"])
        
        if not results["ids"]:
            return []
        
        documents_data = []
        for i, (doc_id, content, metadata) in enumerate(zip(results["ids"], results["documents"], results["metadatas"])):
            documents_data.append({
                "ID": doc_id,
                "Konten": content[:100] + "..." if len(content) > 100 else content,
                "BAB": metadata.get("bab_nomor", "N/A"),
                "Pasal": metadata.get("pasal_nomor", "N/A"),
                "Token": metadata.get("jumlah_token", "N/A"),
                "Source": metadata.get("source", "Original"),
                "Doc Type": metadata.get("doc_type", "original")
            })
        
        return documents_data
    except Exception as e:
        st.error(f"Error mengambil dokumen: {str(e)}")
        return []
def preprocess_new_document(text, doc_name="Dokumen Baru"):
    """Memproses dokumen baru dengan preprocessing yang sama"""
    # Gunakan fungsi preprocessing yang sudah ada
    preprocessed_text = preprocess_text_for_indexing(text)
    
    # Chunk dokumen
    documents = chunk_uu_by_token_for_new_doc(preprocessed_text,  doc_name = doc_name)
    
    # Update metadata untuk dokumen baru
    for doc in documents:
        doc.metadata["source"] = doc_name
        doc.metadata["doc_type"] = "user_upload"
    
    return documents

# Enhanced preprocessing function with step-by-step tracking
def preprocess_query_with_steps(query):
    """
    Preprocess query with step-by-step tracking for display
    """
    steps = []
    
    # Step 1: Original query
    steps.append({
        "step": "1. Query Asli",
        "description": "Query input dari pengguna",
        "before": query,
        "after": query,
        "changes": "Tidak ada perubahan"
    })
    
    # Step 3: Punctuation removal
    query_punctuation = query.translate(str.maketrans("", "", string.punctuation))
    steps.append({
        "step": "2. Penghapusan Tanda Baca",
        "description": "Menghapus semua tanda baca (!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~)",
        "before": query,
        "after": query_punctuation,
        "changes": f"Dihapus: {set(query) - set(query_punctuation)}" if query != query_punctuation else "Tidak ada tanda baca"
    })

    # Step 2: Case folding (lowercase)
    query_final = query_punctuation.lower()
    steps.append({
        "step": "3. Case Folding",
        "description": "Mengubah semua huruf menjadi huruf kecil",
        "before": query_punctuation,
        "after": query_final,
        "changes": "Semua huruf kapital → huruf kecil" if query != query_final else "Tidak ada perubahan"
    })
    
    
    return query_final, steps

# Display preprocessing steps in a nice format
def display_preprocessing_steps(steps):
    """Display preprocessing steps in an organized manner"""
    st.markdown("### 🔄 Proses Preprocessing Query")
    
    # Create expandable section
    with st.expander("Lihat Detail Preprocessing", expanded=True):
        for i, step in enumerate(steps):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"**{step['step']}**")
                st.caption(step['description'])
                
            with col2:
                # Before/After comparison
                if step['before'] != step['after']:
                    st.markdown(f"**Sebelum:** `{step['before']}`")
                    st.markdown(f"**Sesudah:** `{step['after']}`")
                    st.success(f"✅ {step['changes']}")
                else:
                    st.markdown(f"**Teks:** `{step['before']}`")
                    st.info(f"ℹ️ {step['changes']}")
            
            if i < len(steps) - 1:
                st.divider()
    
    # Summary box
    original_query = steps[0]['after']
    final_query = steps[-1]['after']
    
    st.markdown("#### 📋 Ringkasan Preprocessing")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Query Asli:**")
        st.code(original_query, language="text")
        
    with col2:
        st.markdown("**Query Setelah Preprocessing:**")
        st.code(final_query, language="text")
    
    # Show character changes
    if original_query != final_query:
        changes_made = []
        if original_query.lower() != original_query:
            changes_made.append("Case folding")
        if any(c in string.punctuation for c in original_query):
            changes_made.append("Penghapusan tanda baca")
        if original_query != original_query.strip():
            changes_made.append("Normalisasi spasi")
            
        st.success(f"**Perubahan yang diterapkan:** {', '.join(changes_made)}")
    else:
        st.info("**Tidak ada perubahan pada query**")

# Build hybrid retriever (unchanged)
def build_hybrid_retriever(embedding_model, tfidf_k=2, embed_k=2, weights=[0.8, 0.2], persist_directory=DEFAULT_PERSIST_DIR):
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    raw_docs = db.get(include=["documents", "metadatas"])
    documents = [Document(page_content=doc, metadata=meta) for doc, meta in zip(raw_docs["documents"], raw_docs["metadatas"])]

    tfidf_retriever = TFIDFRetriever.from_documents(documents)
    tfidf_retriever.k = tfidf_k

    embedding_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": embed_k})
    hybrid_retriever = EnsembleRetriever(retrievers=[embedding_retriever, tfidf_retriever], weights=weights, k=2)

    return hybrid_retriever

# Simplified calculation functions for tab integration
def compute_tfidf_similarity_for_tab(query, tfidf_results):
    """Compute TF-IDF similarity for tab display"""
    tfidf_vectorizer = TfidfVectorizer()
    docs_content = [query] + [doc.page_content for doc in tfidf_results]
    docs_tfidf = tfidf_vectorizer.fit_transform(docs_content)
    
    # Get query terms for display
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_array = docs_tfidf.toarray()
    
    query_terms = query.lower().split()
    query_feature_indices = []
    query_features = []
    
    for term in query_terms:
        if term in feature_names:
            idx = list(feature_names).index(term)
            query_feature_indices.append(idx)
            query_features.append(term)
    
    # Create matrix for display
    matrix_data = None
    if query_feature_indices:
        query_tfidf_values = tfidf_array[:, query_feature_indices]
        matrix_data = pd.DataFrame(query_tfidf_values, 
                                 index=['Query'] + [f'Doc {i+1}' for i in range(len(tfidf_results))],
                                 columns=query_features)
    else:
        query_vector = tfidf_array[0]
        top_indices = query_vector.argsort()[-5:][::-1]
        matrix_data = pd.DataFrame({
            'Term': [feature_names[i] for i in top_indices],
            'TF-IDF Score': [query_vector[i] for i in top_indices]
        })
    
    # Calculate similarities
    tfidf_similarities = cosine_similarity(docs_tfidf[0:1], docs_tfidf[1:]).flatten()
    tfidf_scores = {doc.page_content: float(score) for doc, score in zip(tfidf_results, tfidf_similarities)}
    
    return tfidf_scores, matrix_data, tfidf_similarities

def compute_embedding_similarity_for_tab(query, embed_results, embedding_model):
    """Compute embedding similarity for tab display"""
    embed_query = embedding_model.embed_query(query)
    embed_docs = [embedding_model.embed_query(doc.page_content) for doc in embed_results]
    
    # Vector information with sample vector display
    def format_vector_sample(vector):
        """Format vector to show first 5 and last 1 elements"""
        if len(vector) > 6:
            first_5 = [round(x, 3) for x in vector[:5]]
            last_1 = round(vector[-1], 3)
            return f"[{', '.join(map(str, first_5))}, ..., {last_1}]"
        else:
            return f"[{', '.join([str(round(x, 3)) for x in vector])}]"
    
    embedding_info = pd.DataFrame({
        'Item': ['Query'] + [f'Doc {i+1}' for i in range(len(embed_results))],
        'Vector Length': [len(embed_query)] + [len(vec) for vec in embed_docs],
        'Vector Sample': [format_vector_sample(embed_query)] + [format_vector_sample(vec) for vec in embed_docs],
        # 'L2 Norm': [round(np.linalg.norm(embed_query), 4)] + [round(np.linalg.norm(vec), 4) for vec in embed_docs]
    })
    
    # Calculate similarities
    embed_similarities = cosine_similarity([embed_query], embed_docs).flatten()
    embed_scores = {doc.page_content: float(score) for doc, score in zip(embed_results, embed_similarities)}
    
    return embed_scores, embedding_info, embed_similarities

def compute_rrf_for_tab(embed_results, tfidf_results, weights=[0.7, 0.3], k=60):
    """Compute RRF scores for tab display"""
    all_docs = list(set([doc.page_content for doc in embed_results + tfidf_results]))
    
    rrf_calculation_data = []
    for idx, doc_content in enumerate(all_docs):
        # Find ranks
        embed_rank = None
        for i, doc in enumerate(embed_results):
            if doc.page_content == doc_content:
                embed_rank = i + 1
                break
        
        tfidf_rank = None
        for i, doc in enumerate(tfidf_results):
            if doc.page_content == doc_content:
                tfidf_rank = i + 1
                break
        
        # Calculate RRF scores
        embed_rrf = 1 / (embed_rank + k) if embed_rank else 0
        tfidf_rrf = 1 / (tfidf_rank + k) if tfidf_rank else 0
        combined_rrf = weights[0] * embed_rrf + weights[1] * tfidf_rrf
        
        rrf_calculation_data.append({
            'Doc': f'Doc {idx + 1}',
            'Embed Rank': embed_rank if embed_rank is not None else np.nan,
            'TF-IDF Rank': tfidf_rank if tfidf_rank is not None else np.nan,
            'Embed RRF': f'{embed_rrf:.4f}',
            'TF-IDF RRF': f'{tfidf_rrf:.4f}',
            'Final RRF': f'{combined_rrf:.4f}',
            'Content': doc_content
        })
    
    rrf_df = pd.DataFrame(rrf_calculation_data)
    rrf_df = rrf_df.sort_values('Final RRF', ascending=False)
    
    # Return RRF scores
    final_rrf_embed = {doc.page_content: 1 / (rank + 1 + k) for rank, doc in enumerate(embed_results)}
    final_rrf_tfidf = {doc.page_content: 1 / (rank + 1 + k) for rank, doc in enumerate(tfidf_results)}
    
    return final_rrf_embed, final_rrf_tfidf, rrf_df

# Simplified context retrieval without detailed display
def get_retriever_context(input_query):
    retriever = build_hybrid_retriever(embedding_model=embeddings)
    tfidf_results = retriever.retrievers[1].invoke(input_query)
    embed_results = retriever.retrievers[0].invoke(input_query)
    final_results = retriever.invoke(input_query)

    # Calculate scores for later use
    tfidf_scores, _, _ = compute_tfidf_similarity_for_tab(input_query, tfidf_results)
    embed_scores, _, _ = compute_embedding_similarity_for_tab(input_query, embed_results, embeddings)
    rrf_embed, rrf_tfidf, _ = compute_rrf_for_tab(embed_results, tfidf_results)
    
    # Final combined scores
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
        "query": input_query,
        "tfidf_results": tfidf_results,
        "embed_results": embed_results,
        "final_results": final_results
    }

# Enhanced display function with calculations integrated in tabs
def tampilkan_hasil_dengan_perhitungan(context_data):
    if not context_data.get("chunks"):
        return
    
    st.markdown("### Retrieved Documents")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Final Results (RFF)", "Sparse Retrieval Results (TF-IDF)", "Dense Retrieval Results (Embedding)"])
    
    # Tab 1: Final Results with RRF calculations
    with tab1:
       
        st.markdown("#### Perhitungan Reciprocal Rank Fusion (RRF)")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info(f"**RRF Formula:**\n\nRRF(d) = Σ(w_i × 1/(rank_i(d) + k))\n\n**Parameters:**\n- k = 60\n- weights = [0.7, 0.3]")
        
        with col2:
            _, _, rrf_df = compute_rrf_for_tab(
                context_data["embed_results"], 
                context_data["tfidf_results"]
            )
            st.dataframe(rrf_df.drop('Content', axis=1), use_container_width=True)
        
        st.divider()
        
        # Display final documents
        for i, chunk in enumerate(context_data["chunks"]):
            with st.container():
                st.markdown(f"**Document {i+1}**")
                
                # Metrics in columns
                col1, col2, col3, = st.columns(3)
                
                with col1:
                    hybrid_score = context_data["scores"].get(chunk, 0)
                    st.metric("RRF Score", f"{hybrid_score:.4f}")
                
                with col2:
                    tfidf_score = context_data["tfidf_scores"].get(chunk, 0)
                    st.metric("TF-IDF", f"{tfidf_score:.4f}")
                
                with col3:
                    embed_score = context_data["embed_scores"].get(chunk, 0)
                    st.metric("Embedding", f"{embed_score:.4f}")
                
                
                # Document content
                with st.expander(f"View Document {i+1} Content", expanded=False):
                    st.text_area(" ", chunk, height=150, disabled=True, key=f"final_{i}",  label_visibility="collapsed" )
                
                st.divider()
    
    # Tab 2: TF-IDF Results with calculations
    with tab2:
       
        st.markdown("#### Perhitungan Sparse Retrieval (TF-IDF)")
        
        tfidf_scores, matrix_data, similarities = compute_tfidf_similarity_for_tab(
            context_data["query"], 
            context_data["tfidf_results"]
        )
        

        st.markdown("**TF-IDF Matrix:**")
        st.dataframe(matrix_data.round(4), use_container_width=True)
        
        st.markdown("**Similarity Scores:**")
        tfidf_sim_df = pd.DataFrame({
            'Document': [f'Doc {i+1}' for i in range(len(context_data["tfidf_results"]))],
            'Similarity': similarities.round(4),
            'Preview': [doc.page_content[:80] + '...' for doc in context_data["tfidf_results"]]
        })
        st.dataframe(tfidf_sim_df, use_container_width=True)
        
        st.divider()
        
        # Display TF-IDF documents
        for i, chunk in enumerate(context_data["tfidf_chunks"]):
            with st.container():
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    tfidf_score = context_data["tfidf_scores"].get(chunk, 0)
                    st.metric(f"TF-IDF Rank {i+1}", f"{tfidf_score:.4f}")
                
                with col2:
                    with st.expander(f"TF-IDF Document {i+1}", expanded=False):
                        st.text_area(" ", chunk, height=100, disabled=True, key=f"tfidf_{i}",  label_visibility="collapsed" )
    
    # Tab 3: Embedding Results with calculations
    with tab3:
        
        st.markdown("#### Perhitungan Dense Retrieval (text-embedding-3-small)")
        
        embed_scores, embedding_info, similarities = compute_embedding_similarity_for_tab(
            context_data["query"], 
            context_data["embed_results"], 
            embeddings
        )
        
        
        st.markdown("**Vector Information:**")
        st.dataframe(embedding_info, use_container_width=True)
        
        # Similarity scores
        st.markdown("**Similarity Scores:**")
        embed_sim_df = pd.DataFrame({
            'Document': [f'Doc {i+1}' for i in range(len(context_data["embed_results"]))],
            'Similarity': similarities.round(4),
            'Preview': [doc.page_content[:80] + '...' for doc in context_data["embed_results"]]
        })
        st.dataframe(embed_sim_df, use_container_width=True)
    
        st.divider()
        
        # Display embedding documents
        for i, chunk in enumerate(context_data["embed_chunks"]):
            with st.container():
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    embed_score = context_data["embed_scores"].get(chunk, 0)
                    st.metric(f"Embed Rank {i+1}", f"{embed_score:.4f}")
                
                with col2:
                    with st.expander(f"Embedding Document {i+1}", expanded=False):
                        st.text_area(" ", chunk, height=100, disabled=True, key=f"embed_{i}",  label_visibility="collapsed" )

# RAG and non-RAG response generators with preprocessing
def stream_rag_response(query):
    # Preprocess query
    processed_query, preprocessing_steps = preprocess_query_with_steps(query)
    
    llm = OllamaLLM(model="qwen3:8b", temperature=0)
    context_data = get_retriever_context(processed_query)
    context = context_data["context"]
    
    # Add preprocessing info to context_data
    context_data["preprocessing_steps"] = preprocessing_steps
    context_data["original_query"] = query
    context_data["processed_query"] = processed_query

    template = """<|im_start|>system
            Kamu adalah asisten hukum spesialis ketenagakerjaan di Indonesia. Tugasmu adalah menjawab pertanyaan pengguna secara singkat dan hanya berdasarkan kutipan resmi dari Undang-Undang yang tersedia dalam konteks.

            Aturan ketat dalam menjawab:
            - Jawaban diawali dengan penjelasan kemudian diikuti dengan kutipan pasal/ayat yang relevan berdasarkan konteks yang diberikan.
            - Jika ada nomor dan list, tolong gunakan format yang sesuai.
            - Jawaban hanya boleh berdasarkan konteks yang tersedia.
            - Jika pasal dalam konteks telah diubah, tambahkan keterangan bahwa isi tersebut merupakan hasil amandemen dan sebutkan UU yang mengubahnya.
            - Jika pasal telah dihapus, sebutkan bahwa pasal tersebut sudah tidak berlaku dan tidak perlu dijelaskan lebih lanjut.
            - Sebutkan pasal dan ayat secara eksplisit jika tersedia.
            - Jangan menambahkan interpretasi atau opini di luar kutipan.
            - Jika tidak ditemukan informasi yang relevan dalam konteks, jawab: "Maaf saya tidak bisa menjawab dikarenakan informasi tidak tersedia di dalam dokumen"

            Format konteks:
            - konteks terdiri dari beberapa sumber hukum yang dituliskan pada awal konteks.
            - Dimulai dengan nama dokumen ,bab dengan contoh: uu nomor 13 tahun 2003 ketenagakerjaan, bab ix Hubungan Kerja.
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
            Berikut adalah konteks dari dokumen hukum yang relevan:

            {context}

            Pertanyaan:
            (Jawab hanya jika ditemukan kutipan pasal yang relevan dalam kontek dan sertakan sumber hukumnya)
            {question}
            <|im_end|>
            <|im_start|>assistant
            /no_think
            """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.stream({"context": context, "question": processed_query}), context_data

def get_no_rag_response(query):
    # Preprocess query for no-RAG version too
    processed_query, _ = preprocess_query_with_steps(query)
    
    llm = OllamaLLM(model="qwen3:8b", temperature=0.1)
    template = """
        <|im_start|>system
        Kamu asisten hukum yang hanya menjawab berdasarkan peraturan ketenagakerjaan Indonesia khususnya Undang-Undang Nomor 13 Tahun 2003 tentang Ketenagakerjaan beserta perubahannya, termasuk dari UU No. 6 Tahun 2023 (Cipta Kerja). Jawab singkat, sebutkan pasal/ayat dan jelaskan isinya. Jika pasal hasil amandemen, beri tahu.
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
    return chain.stream({"question": processed_query})

# Enhanced Streamlit UI with better layout
st.set_page_config(
    page_title="QA UU Ketenagakerjaan", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .calculation-section {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .preprocessing-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("Navigasi")

    menu_utama = st.radio(
        "Pilih Halaman:",
        ["Tanya Jawab", "Kelola Data"],
        help="Navigasi utama"
    )

    st.divider()

    if menu_utama == "Tanya Jawab":
        mode = st.radio(
            "Mode Jawaban", 
            ["rag", "no_rag", "both"], 
            format_func=lambda x: {
                "rag": "Dengan RAG", 
                "no_rag": "Tanpa RAG", 
                "both": "Keduanya"
            }[x],
            help="Pilih mode untuk mendapatkan jawaban"
        )

        # show_preprocessing = st.checkbox("Tampilkan Preprocessing Query", value=True)
        # show_calculations = st.checkbox("Tampilkan Proses Perhitungan", value=True)
        # show_documents = st.checkbox("Tampilkan Dokumen Lengkap", value=False)

    elif menu_utama == "Kelola Data":
        db_action = st.selectbox(
            "Pilih Aksi:",
            ["Lihat Database", "Tambah Data", "Hapus Data"],
            help="Pilih aksi untuk mengelola database Chroma"
        )

if menu_utama == "Kelola Data":
    if db_action == "Lihat Database":
        st.markdown("### 📋 Dokumen dalam Database")
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("🔄 Refresh Database", use_container_width=True):
                st.rerun()
        
        with col2:
            st.empty()
        
        # Load and display documents
        docs_data = view_all_documents()
        if docs_data:
            df = pd.DataFrame(docs_data)
            
            # Filter options
            with st.expander("🔍 Filter Dokumen", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    doc_types = df['Doc Type'].unique().tolist()
                    selected_doc_type = st.multiselect("Filter by Doc Type:", doc_types, default=doc_types)
                
                with col2:
                    bab_list = df['BAB'].unique().tolist()
                    selected_bab = st.multiselect("Filter by BAB:", bab_list, default=bab_list)
                
                with col3:
                    sources = df['Source'].unique().tolist()
                    selected_sources = st.multiselect("Filter by Source:", sources, default=sources)
            
            # Apply filters
            filtered_df = df[
                (df['Doc Type'].isin(selected_doc_type)) &
                (df['BAB'].isin(selected_bab)) &
                (df['Source'].isin(selected_sources))
            ]
            
            st.dataframe(filtered_df, use_container_width=True)
            st.info(f"Menampilkan {len(filtered_df)} dari {len(docs_data)} total dokumen")
        else:
            st.warning("Database kosong atau terjadi error.")

    elif db_action == "Tambah Data":
        st.markdown("### ➕ Tambah Data Baru")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload file (Peraturan Ketenagakerjaan):",
            type=['pdf'],
            help="Upload dokumen hukum yang akan ditambahkan ke database"
        )
        
        
        # Nama dokumen
        doc_name = st.text_input(
            "Nama dokumen:",
            placeholder="Contoh: UU No. 14 Tahun 2005"
        )
        
        # Preview preprocessing
        show_preview = st.checkbox("Tampilkan preview preprocessing", value=True)
        
        if st.button("📥 Tambah ke Database", type="primary", use_container_width=True):
            if not doc_name:
                st.error("Nama dokumen harus diisi!")
            elif uploaded_file is not None:
                with st.spinner("Memproses file..."):
                    # Proses file upload
                    raw_text = process_uploaded_file(uploaded_file)
                    if raw_text:
                        # Preprocessing dan chunking
                        documents = preprocess_new_document(raw_text, doc_name)
                        
                        # Tampilkan preview
                        if show_preview:
                            st.write(f"📄 Preview preprocessing (3 chunk pertama dari {len(documents)} total):")
                            for i, doc in enumerate(documents[:3]):
                                with st.expander(f"Chunk {i+1}", expanded=False):
                                    col1, col2 = st.columns([1, 2])
                                    with col1:
                                        st.json(doc.metadata)
                                    with col2:
                                        st.text_area("Konten:", doc.page_content, height=150, disabled=True, key=f"preview_{i}")
                        
                        # Tambah ke database
                        success, new_ids = add_documents_to_chroma(documents, embeddings)
                        if success:
                            st.balloons()
                            st.success(f"✅ Berhasil menambahkan {len(documents)} chunks dengan ID: {new_ids[:3]}...")
            else:
                st.error("Silakan upload file!")

    elif db_action == "Hapus Data":
        st.markdown("### 🗑️ Hapus Data")
        
        # Tampilkan dokumen yang bisa dihapus
        docs_data = view_all_documents()
        
        if docs_data:
            df = pd.DataFrame(docs_data)
            sources = df['Source'].unique().tolist()
            selected_source = st.selectbox("Pilih Source untuk dihapus:", sources)
            
            if selected_source:
                selected_df = df[df['Source'] == selected_source]
                selected_ids = selected_df['ID'].tolist()
                st.info(f"Akan menghapus {len(selected_ids)} dokumen dari source: {selected_source}")
            else:
                selected_ids = []
            
           
            if selected_ids:
                # Tampilkan preview dokumen yang akan dihapus
                st.write("**⚠️ Preview dokumen yang akan dihapus:**")
                preview_df = df[df['ID'].isin(selected_ids)]
                st.dataframe(preview_df, use_container_width=True)
                
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if st.button("🗑️ Hapus Dokumen", type="secondary", use_container_width=True):
                        if delete_documents_from_chroma(selected_ids):
                            st.success("✅ Dokumen berhasil dihapus!")
                            st.rerun()
            
                with col2:
                    if st.button("❌ Batal", use_container_width=True):
                        st.rerun()
            
            else:
                st.info("Pilih dokumen yang akan dihapus.")
        else:
            st.info("Tidak ada dokumen dalam database.")


elif menu_utama == "Tanya Jawab":
     # ========= SEPARATOR UNTUK INTERFACE UTAMA =========
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🏛️ Tanya Jawab UU Ketenagakerjaan")
    st.markdown('</div>', unsafe_allow_html=True)
    # Query input
    query = st.text_input(
        "Masukkan pertanyaan Anda:",
        placeholder="Contoh: Apa saja hak pekerja menurut UU Ketenagakerjaan?",
        help="Ketik pertanyaan tentang UU Ketenagakerjaan Indonesia"
    )

    if st.button("Tanyakan", type="primary", use_container_width=True) and query:
        # Hilangkan tanda tanya di akhir query jika ada
        # query = query.lower().translate(str.maketrans("", "", string.punctuation)).strip().strip()
        
        processed_query, preprocessing_steps = preprocess_query_with_steps(query)
        display_preprocessing_steps(preprocessing_steps)
        st.markdown("---")
        
        # Use original preprocessing logic for actual processing
        final_query = query.lower().translate(str.maketrans("", "", string.punctuation)).strip()

        if mode in ["rag"]:
            st.markdown("## Jawaban dengan RAG")
            
            with st.spinner("Menganalisis dan mengambil konteks..."):
                # Get RAG response
                rag_stream, context_data = stream_rag_response(query)
                
                # Display answer
                # st.markdown('<div class="answer-container">', unsafe_allow_html=True)
                st.markdown("### Jawaban:")
                answer_placeholder = st.empty()
                output = ""
                for token in rag_stream:
                    output += token
                    clean_output = re.sub(r"</?think>", "", output)
                    answer_placeholder.markdown(clean_output)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show documents with integrated calculations
                
                st.markdown("---")
                tampilkan_hasil_dengan_perhitungan(context_data)
        
        if mode in ["no_rag"]:
            if mode == "both":
                st.markdown("---")
            
            st.markdown("## Jawaban tanpa RAG")
            
            with st.spinner(" Menghasilkan jawaban..."):
                # st.markdown('<div class="answer-container">', unsafe_allow_html=True)
                st.markdown("### Jawaban:")
                no_rag_placeholder = st.empty()
                no_rag_stream = get_no_rag_response(query)
                output = ""
                for token in no_rag_stream:
                    output += token
                    clean_output = re.sub(r"</?think>", "", output)
                    no_rag_placeholder.markdown(clean_output)
                st.markdown('</div>', unsafe_allow_html=True)
