import os
import re
import pdfplumber
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL
import pandas as pd
import networkx as nx
import mat*plotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Download resources yang diperlukan
nltk.download('punkt', quiet=True)

# ---------- BAGIAN 1: PERSIAPAN ----------

def setup_directories():
    """Membuat direktori yang diperlukan untuk proses konversi"""
    directories = ['data', 'output', 'temp']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
    return "Direktori berhasil dibuat"

# ---------- BAGIAN 2: EKSTRAKSI TEKS DARI PDF ----------

def extract_text_from_pdf(pdf_path):
    """
    Mengekstrak teks dari file PDF
    Args:
        pdf_path: Path ke file PDF
    Returns:
        String berisi teks yang diekstrak
    """
    all_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in tqdm(pdf.pages, desc="Mengekstrak halaman"):
                text = page.extract_text()
                if text:
                    all_text += text + "\n\n"
        print(f"Berhasil mengekstrak {len(pdf.pages)} halaman dari PDF")
        return all_text
    except Exception as e:
        print(f"Terjadi kesalahan saat mengekstrak PDF: {e}")
        return None

# ---------- BAGIAN 3: PREPROCESSING TEKS ----------

def preprocess_text(text):
    """
    Melakukan preprocessing pada teks yang diekstrak
    Args:
        text: Teks yang akan dipreprocess
    Returns:
        Teks hasil preprocessing
    """
    if not text:
        return ""
    
    # Menghapus header dan footer yang berulang
    cleaned_text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Menghapus karakter non-printable
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', cleaned_text)
    
    # Menghapus spasi berlebih
    cleaned_text = re.sub(r' +', ' ', cleaned_text)
    
    return cleaned_text

# ---------- BAGIAN 4: SEGMENTASI DOKUMEN ----------

def segment_document(text):
    """
    Mensegmentasi dokumen menjadi bagian-bagian seperti Menimbang, Mengingat, Bab, Pasal, dll
    Args:
        text: Teks dokumen yang sudah dipreprocess
    Returns:
        Dictionary berisi segmen-segmen dokumen
    """
    segments = {
        'header': '',
        'menimbang': '',
        'mengingat': '',
        'memutuskan': '',
        'bab': [],
        'pasal': []
    }
    
    # Mengekstrak bagian Menimbang
    menimbang_match = re.search(r'Menimbang\s*:([^M]*?)(?=Mengingat|$)', text, re.IGNORECASE | re.DOTALL)
    if menimbang_match:
        segments['menimbang'] = menimbang_match.group(1).strip()
    
    # Mengekstrak bagian Mengingat
    mengingat_match = re.search(r'Mengingat\s*:([^M]*?)(?=MEMUTUSKAN|$)', text, re.IGNORECASE | re.DOTALL)
    if mengingat_match:
        segments['mengingat'] = mengingat_match.group(1).strip()
    
    # Mengekstrak bagian header (judul peraturan)
    header_match = re.search(r'^(.*?)(?=Menimbang|MENIMBANG)', text, re.IGNORECASE | re.DOTALL)
    if header_match:
        segments['header'] = header_match.group(1).strip()
    
    # Mengekstrak bagian Memutuskan
    memutuskan_match = re.search(r'(?:MEMUTUSKAN|Memutuskan)\s*:([^B]*?)(?=BAB|Pasal|$)', text, re.IGNORECASE | re.DOTALL)
    if memutuskan_match:
        segments['memutuskan'] = memutuskan_match.group(1).strip()
    
    # Mengekstrak Bab
    bab_matches = re.finditer(r'BAB\s+([IVX]+)\s+(.*?)(?=BAB\s+[IVX]+|\Z)', text, re.IGNORECASE | re.DOTALL)
    for match in bab_matches:
        bab_number = match.group(1)
        bab_title = match.group(2).strip()
        bab_content = match.group(0)
        segments['bab'].append({
            'number': bab_number,
            'title': bab_title,
            'content': bab_content
        })
    
    # Mengekstrak Pasal
    pasal_matches = re.finditer(r'Pasal\s+(\d+)([^P]*?)(?=Pasal\s+\d+|\Z)', text, re.IGNORECASE | re.DOTALL)
    for match in pasal_matches:
        pasal_number = match.group(1)
        pasal_content = match.group(2).strip()
        segments['pasal'].append({
            'number': pasal_number,
            'content': pasal_content
        })
    
    return segments

# ---------- BAGIAN 5: EKSTRAKSI INFORMASI METADATA ----------

def extract_metadata(text, segments):
    """
    Mengekstrak metadata dari dokumen legal
    Args:
        text: Teks dokumen lengkap
        segments: Segmen-segmen dokumen
    Returns:
        Dictionary berisi metadata dokumen
    """
    metadata = {
        'jenis_peraturan': None,
        'nomor': None,
        'tahun': None,
        'tentang': None,
        'tanggal_disahkan': None,
        'disahkan_oleh': None,
        'jabatan_pengesah': None,
        'tempat_disahkan': None,
        'bahasa': 'id'
    }
    
    # Ekstrak jenis peraturan dan nomor
    jenis_nomor_match = re.search(r'(UNDANG-UNDANG|PERATURAN PEMERINTAH|PERATURAN PRESIDEN|PERATURAN MENTERI|PERATURAN DAERAH)\s+(?:REPUBLIK INDONESIA\s+)?(?:NOMOR|NO[.]?)\s+(\d+[-/]?[A-Z]*)\s+TAHUN\s+(\d{4})', text, re.IGNORECASE)
    if jenis_nomor_match:
        jenis = jenis_nomor_match.group(1).upper()
        if 'UNDANG-UNDANG' in jenis:
            metadata['jenis_peraturan'] = 'UU'
        elif 'PERATURAN PEMERINTAH' in jenis:
            metadata['jenis_peraturan'] = 'PP'
        elif 'PERATURAN PRESIDEN' in jenis:
            metadata['jenis_peraturan'] = 'Perpres'
        elif 'PERATURAN MENTERI' in jenis:
            metadata['jenis_peraturan'] = 'PeraturanMenteri'
        elif 'PERATURAN DAERAH' in jenis and 'PROVINSI' in text:
            metadata['jenis_peraturan'] = 'PerdaProvinsi'
        elif 'PERATURAN DAERAH' in jenis:
            metadata['jenis_peraturan'] = 'PerdaKabKota'
        
        metadata['nomor'] = jenis_nomor_match.group(2)
        metadata['tahun'] = jenis_nomor_match.group(3)
    
    # Ekstrak tentang (judul peraturan)
    tentang_match = re.search(r'TENTANG\s+(.*?)(?=\n\n|\nMenimbang|\nDengan|$)', text, re.IGNORECASE | re.DOTALL)
    if tentang_match:
        metadata['tentang'] = tentang_match.group(1).strip()
    
    # Ekstrak tanggal disahkan
    tanggal_match = re.search(r'Ditetapkan di.*\n.*pada tanggal\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})', text, re.IGNORECASE)
    if tanggal_match:
        metadata['tanggal_disahkan'] = tanggal_match.group(1)
    
    # Ekstrak lokasi disahkan
    tempat_match = re.search(r'Ditetapkan di\s+([A-Za-z ]+)', text, re.IGNORECASE)
    if tempat_match:
        metadata['tempat_disahkan'] = tempat_match.group(1).strip()
    
    # Ekstrak pengesah
    pengesah_match = re.search(r'(?:PRESIDEN REPUBLIK INDONESIA|MENTERI|GUBERNUR|BUPATI|WALIKOTA)[,]?\s*\n\s*(?:ttd[.]?\s*\n\s*)?((?:[A-Z][A-Za-z.]+\s*)+)', text)
    if pengesah_match:
        metadata['disahkan_oleh'] = pengesah_match.group(1).strip()
    
    # Ekstrak jabatan pengesah
    jabatan_match = re.search(r'(PRESIDEN REPUBLIK INDONESIA|MENTERI[A-Z\s]+|GUBERNUR[A-Z\s]+|BUPATI[A-Z\s]+|WALIKOTA[A-Z\s]+)[,]?\s*\n', text)
    if jabatan_match:
        metadata['jabatan_pengesah'] = jabatan_match.group(1).strip()
    
    return metadata

# ---------- BAGIAN 6: EKSTRAKSI STRUKTUR PASAL ----------

def extract_article_structure(segments):
    """
    Mengekstrak struktur pasal, ayat, huruf
    Args:
        segments: Segmen-segmen dokumen
    Returns:
        List berisi struktur pasal yang terdeteksi
    """
    articles = []
    
    for pasal in segments['pasal']:
        article = {
            'number': pasal['number'],
            'content': pasal['content'],
            'ayat': [],
            'references': []
        }
        
        # Ekstrak ayat-ayat
        ayat_matches = re.finditer(r'\((\d+)\)(.*?)(?=\(\d+\)|\Z)', pasal['content'], re.DOTALL)
        for ayat_match in ayat_matches:
            ayat_number = ayat_match.group(1)
            ayat_content = ayat_match.group(2).strip()
            
            ayat = {
                'number': ayat_number,
                'content': ayat_content,
                'huruf': []
            }
            
            # Ekstrak huruf (point)
            huruf_matches = re.finditer(r'([a-z])[.)]([^a-z.][^a-z)]*?)(?=[a-z][.)]|\Z)', ayat_content, re.DOTALL)
            for huruf_match in huruf_matches:
                huruf_letter = huruf_match.group(1)
                huruf_content = huruf_match.group(2).strip()
                
                ayat['huruf'].append({
                    'letter': huruf_letter,
                    'content': huruf_content
                })
            
            article['ayat'].append(ayat)
        
        # Jika tidak ada ayat yang terdeteksi, mungkin pasal langsung berisi teks
        if len(article['ayat']) == 0 and article['content']:
            article['text'] = article['content'].strip()
        
        articles.append(article)
    
    return articles

# ---------- BAGIAN 7: EKSTRAKSI REFERENSI ANTAR PASAL ----------

def extract_references(articles):
    """
    Mengekstrak referensi antar pasal
    Args:
        articles: List struktur pasal
    Returns:
        List pasal yang diperkaya dengan informasi referensi
    """
    for article in articles:
        content = article['content']
        
        # Mencari referensi ke pasal lain
        ref_matches = re.finditer(r'[Pp]asal\s+(\d+)', content)
        for match in ref_matches:
            ref_pasal = match.group(1)
            if ref_pasal != article['number']:  # Bukan referensi ke diri sendiri
                article['references'].append({
                    'type': 'merujuk',
                    'target': f"Pasal {ref_pasal}"
                })
    
    return articles

# ---------- BAGIAN 8: KONVERSI KE RDF MENGGUNAKAN ONTOLOGI ----------

def create_knowledge_graph(metadata, segments, articles):
    """
    Membuat knowledge graph dari dokumen legal dalam format RDF
    Args:
        metadata: Metadata dokumen
        segments: Segmen-segmen dokumen
        articles: Struktur pasal yang diekstrak
    Returns:
        Graph RDF
    """
    # Inisialisasi graph RDF
    g = Graph()
    
    # Menambahkan namespace
    lex2kg = Namespace("https://example.org/lex2kg/ontology/")
    schema = Namespace("https://schema.org/")
    doc_ns = Namespace(f"https://example.org/legal-doc/{metadata['jenis_peraturan']}/{metadata['nomor']}/{metadata['tahun']}/")
    
    # Mendaftarkan namespace
    g.bind("lex2kg-o", lex2kg)
    g.bind("schema", schema)
    g.bind("doc", doc_ns)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    
    # Menambahkan metadata dokumen
    doc_uri = doc_ns["document"]
    g.add((doc_uri, RDF.type, lex2kg.Peraturan))
    
    if metadata['jenis_peraturan']:
        g.add((doc_uri, lex2kg.jenisPeraturan, URIRef(f"https://example.org/lex2kg/ontology/{metadata['jenis_peraturan']}")))
    
    if metadata['nomor']:
        g.add((doc_uri, lex2kg.nomor, Literal(metadata['nomor'])))
    
    if metadata['tahun']:
        g.add((doc_uri, lex2kg.tahun, Literal(metadata['tahun'])))
    
    if metadata['tentang']:
        g.add((doc_uri, lex2kg.tentang, Literal(metadata['tentang'])))
    
    if metadata['tanggal_disahkan']:
        g.add((doc_uri, lex2kg.disahkanPada, Literal(metadata['tanggal_disahkan'])))
    
    if metadata['disahkan_oleh']:
        g.add((doc_uri, lex2kg.disahkanOleh, Literal(metadata['disahkan_oleh'])))
    
    if metadata['jabatan_pengesah']:
        g.add((doc_uri, lex2kg.jabatanPengesah, Literal(metadata['jabatan_pengesah'])))
    
    if metadata['tempat_disahkan']:
        g.add((doc_uri, lex2kg.disahkanDi, Literal(metadata['tempat_disahkan'])))
    
    g.add((doc_uri, lex2kg.bahasa, Literal(metadata['bahasa'])))
    
    # Menambahkan segmen Menimbang
    if segments['menimbang']:
        menimbang_uri = doc_ns["menimbang"]
        g.add((menimbang_uri, RDF.type, lex2kg.Menimbang))
        g.add((menimbang_uri, lex2kg.teks, Literal(segments['menimbang'])))
        g.add((doc_uri, lex2kg.menimbang, menimbang_uri))
    
    # Menambahkan segmen Mengingat
    if segments['mengingat']:
        mengingat_uri = doc_ns["mengingat"]
        g.add((mengingat_uri, RDF.type, lex2kg.Mengingat))
        g.add((mengingat_uri, lex2kg.teks, Literal(segments['mengingat'])))
        g.add((doc_uri, lex2kg.mengingat, mengingat_uri))
    
    # Menambahkan Bab
    bab_list_uri = doc_ns["daftarBab"]
    g.add((bab_list_uri, RDF.type, lex2kg.DaftarBab))
    g.add((doc_uri, lex2kg.daftarBab, bab_list_uri))
    
    for i, bab in enumerate(segments['bab']):
        bab_uri = doc_ns[f"bab/{bab['number']}"]
        g.add((bab_uri, RDF.type, lex2kg.Bab))
        g.add((bab_uri, lex2kg.nomor, Literal(bab['number'])))
        g.add((bab_uri, lex2kg.teks, Literal(bab['title'])))
        g.add((bab_uri, lex2kg.bagianDari, doc_uri))
        g.add((bab_list_uri, lex2kg.bab, bab_uri))
    
    # Menambahkan Pasal
    pasal_list_uri = doc_ns["daftarPasal"]
    g.add((pasal_list_uri, RDF.type, lex2kg.DaftarPasal))
    g.add((doc_uri, lex2kg.daftarPasal, pasal_list_uri))
    
    for article in articles:
        pasal_uri = doc_ns[f"pasal/{article['number']}"]
        g.add((pasal_uri, RDF.type, lex2kg.Pasal))
        g.add((pasal_uri, lex2kg.nomor, Literal(article['number'])))
        
        if 'text' in article:
            g.add((pasal_uri, lex2kg.teks, Literal(article['text'])))
        
        g.add((pasal_uri, lex2kg.bagianDari, doc_uri))
        g.add((pasal_list_uri, lex2kg.pasal, pasal_uri))
        
        # Menambahkan Ayat
        if article['ayat']:
            ayat_list_uri = doc_ns[f"pasal/{article['number']}/daftarAyat"]
            g.add((ayat_list_uri, RDF.type, lex2kg.DaftarAyat))
            g.add((pasal_uri, lex2kg.daftarAyat, ayat_list_uri))
            
            for ayat in article['ayat']:
                ayat_uri = doc_ns[f"pasal/{article['number']}/ayat/{ayat['number']}"]
                g.add((ayat_uri, RDF.type, lex2kg.Ayat))
                g.add((ayat_uri, lex2kg.nomor, Literal(ayat['number'])))
                g.add((ayat_uri, lex2kg.teks, Literal(ayat['content'])))
                g.add((ayat_uri, lex2kg.bagianDari, pasal_uri))
                g.add((ayat_list_uri, lex2kg.ayat, ayat_uri))
                
                # Menambahkan Huruf
                if ayat['huruf']:
                    huruf_list_uri = doc_ns[f"pasal/{article['number']}/ayat/{ayat['number']}/daftarHuruf"]
                    g.add((huruf_list_uri, RDF.type, lex2kg.DaftarHuruf))
                    g.add((ayat_uri, lex2kg.daftarHuruf, huruf_list_uri))
                    
                    for huruf in ayat['huruf']:
                        huruf_uri = doc_ns[f"pasal/{article['number']}/ayat/{ayat['number']}/huruf/{huruf['letter']}"]
                        g.add((huruf_uri, RDF.type, lex2kg.Huruf))
                        g.add((huruf_uri, lex2kg.nomor, Literal(huruf['letter'])))
                        g.add((huruf_uri, lex2kg.teks, Literal(huruf['content'])))
                        g.add((huruf_uri, lex2kg.bagianDari, ayat_uri))
                        g.add((huruf_list_uri, lex2kg.huruf, huruf_uri))
        
        # Menambahkan referensi
        for ref in article['references']:
            if ref['type'] == 'merujuk':
                target_pasal_num = ref['target'].split(' ')[1]
                target_pasal_uri = doc_ns[f"pasal/{target_pasal_num}"]
                g.add((pasal_uri, lex2kg.merujuk, target_pasal_uri))
    
    return g

# ---------- BAGIAN 9: VISUALISASI KNOWLEDGE GRAPH ----------

def visualize_knowledge_graph(g, output_path='output/kg_visualization.png'):
    """
    Memvisualisasikan knowledge graph
    Args:
        g: Graph RDF
        output_path: Path untuk menyimpan visualisasi
    """
    # Konversi RDF Graph ke NetworkX Graph untuk visualisasi
    G = nx.Graph()
    
    # Tambahkan node dan edge
    for s, p, o in g:
        s_label = s.split('/')[-1]
        if isinstance(o, Literal):
            o_label = str(o)[:20] + '...' if len(str(o)) > 20 else str(o)
        else:
            o_label = o.split('/')[-1]
        
        p_label = p.split('/')[-1]
        
        G.add_node(str(s), label=s_label)
        G.add_node(str(o), label=o_label)
        G.add_edge(str(s), str(o), label=p_label)
    
    # Visualisasi dengan NetworkX
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G)
    
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title("Knowledge Graph dari Dokumen Legal")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualisasi disimpan di {output_path}")

# ---------- BAGIAN 10: FUNGSI UTAMA UNTUK MENJALANKAN SELURUH PROSES ----------

def create_legal_knowledge_graph(pdf_path, output_dir='output/'):
    """
    Fungsi utama untuk menjalankan seluruh proses konversi PDF ke knowledge graph
    Args:
        pdf_path: Path ke file PDF
        output_dir: Direktori untuk menyimpan output
    """
    print(f"Memulai proses konversi {pdf_path} ke knowledge graph")
    
    # Setup direktori
    setup_directories()
    
    # 1. Ekstrak teks dari PDF
    print("Mengekstrak teks dari PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)
    if not extracted_text:
        return "Gagal mengekstrak teks dari PDF"
    
    # 2. Preprocessing teks
    print("Melakukan preprocessing teks...")
    cleaned_text = preprocess_text(extracted_text)
    
    # Simpan teks yang sudah dipreprocess
    with open(f"{output_dir}/preprocessed_text.txt", 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    # 3. Segmentasi dokumen
    print("Melakukan segmentasi dokumen...")
    segments = segment_document(cleaned_text)
    
    # 4. Ekstraksi metadata
    print("Mengekstrak metadata dokumen...")
    metadata = extract_metadata(cleaned_text, segments)
    
    # 5. Ekstraksi struktur pasal
    print("Mengekstrak struktur pasal...")
    articles = extract_article_structure(segments)
    
    # 6. Ekstraksi referensi antar pasal
    print("Mengekstrak referensi antar pasal...")
    articles = extract_references(articles)
    
    # 7. Membuat knowledge graph
    print("Membuat knowledge graph...")
    kg = create_knowledge_graph(metadata, segments, articles)
    
    # 8. Simpan knowledge graph dalam format RDF/Turtle
    output_ttl = f"{output_dir}/legal_knowledge_graph.ttl"
    kg.serialize(destination=output_ttl, format='turtle')
    print(f"Knowledge graph disimpan dalam format Turtle di {output_ttl}")
    
    # 9. Simpan metadata dalam format JSON
    import json
    with open(f"{output_dir}/metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # 10. Visualisasi knowledge graph
    print("Memvisualisasikan knowledge graph...")
    visualize_knowledge_graph(kg, f"{output_dir}/knowledge_graph_visualization.png")
    
    # 11. Buat statistik
    print("Membuat statistik knowledge graph...")
    stats = {
        'total_triples': len(kg),
        'total_pasals': len(articles),
        'jenis_peraturan': metadata['jenis_peraturan'],
        'tahun_terbit': metadata['tahun'],
        'tentang': metadata['tentang']
    }
    
    with open(f"{output_dir}/kg_stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print("Proses konversi PDF ke knowledge graph selesai!")
    return f"Knowledge graph berhasil dibuat dengan {stats['total_triples']} triple"

# ---------- CONTOH PENGGUNAAN ----------

if __name__ == "__main__":
    # Contoh penggunaan
    pdf_path = r"D:\KRSIPSI\CODING\regkerjaaan\data\UU Nomor 13 Tahun 2003.pdf"  # Ganti dengan path ke file PDF dokumen legal Anda
    result = create_legal_knowledge_graph(pdf_path)
    print(result)