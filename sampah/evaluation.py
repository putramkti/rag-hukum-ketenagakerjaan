import os
from typing import List, Dict, Any
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset
import pandas as pd

def prepare_evaluation_dataset(questions: List[Dict[str, str]]) -> Dataset:
    """
    Prepare dataset for RAGAS evaluation
    
    Args:
        questions: List of dictionaries containing questions and ground truth answers
        
    Returns:
        Dataset: Dataset formatted for RAGAS evaluation
    """
    # Convert to pandas DataFrame
    df = pd.DataFrame(questions)
    
    # Create dataset
    dataset = Dataset.from_pandas(df)
    
    return dataset

def evaluate_rag_chain(
    chain,
    vectorstore: Chroma,
    test_questions: List[Dict[str, str]],
    metrics: List = None
) -> Dict[str, float]:
    """
    Evaluate RAG chain using RAGAS metrics
    
    Args:
        chain: RAG chain to evaluate
        vectorstore: Vector store used for retrieval
        test_questions: List of test questions with ground truth answers
        metrics: List of RAGAS metrics to use (defaults to standard metrics)
        
    Returns:
        Dict[str, float]: Dictionary of metric scores
    """
    if metrics is None:
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
    
    # Prepare dataset
    dataset = prepare_evaluation_dataset(test_questions)
    
    # Run evaluation
    result = evaluate(
        dataset=dataset,
        metrics=metrics
    )
    
    return result

def create_test_questions() -> List[Dict[str, str]]:
    """
    Create a list of test questions with ground truth answers
    
    Returns:
        List[Dict[str, str]]: List of test questions
    """
    return [
        {
            "question": "Siapa yang bertanggung jawab atas keselamatan dan kesehatan kerja?",
            "answer": "Pengurus dan pengusaha bertanggung jawab atas keselamatan dan kesehatan kerja sesuai dengan Pasal 87 UU No. 13 Tahun 2003.",
            "context": "Pasal 87: (1) Setiap perusahaan wajib menerapkan sistem manajemen keselamatan dan kesehatan kerja yang terintegrasi dengan sistem manajemen perusahaan. (2) Ketentuan mengenai penerapan sistem manajemen keselamatan dan kesehatan kerja sebagaimana dimaksud dalam ayat (1) diatur dengan Peraturan Pemerintah."
        },
        {
            "question": "Apa saja hak-hak pekerja/buruh?",
            "answer": "Hak-hak pekerja/buruh meliputi: 1) Keselamatan dan kesehatan kerja, 2) Perlindungan hukum, 3) Jaminan sosial, 4) Upah yang layak, 5) Cuti, 6) Jaminan hari tua, 7) Kesempatan mengembangkan diri, 8) Kebebasan berserikat, 9) Kesempatan berpartisipasi dalam pengambilan keputusan, 10) Kesempatan berpartisipasi dalam pengambilan keputusan.",
            "context": "Pasal 88: (1) Setiap pekerja/buruh berhak memperoleh perlindungan atas: a. keselamatan dan kesehatan kerja; b. moral dan kesusilaan; dan c. perlakuan yang sesuai dengan harkat dan martabat manusia serta nilai-nilai agama. (2) Untuk melindungi keselamatan pekerja/buruh guna mewujudkan produktivitas kerja yang optimal diselenggarakan upaya keselamatan dan kesehatan kerja. (3) Perlindungan sebagaimana dimaksud dalam ayat (1) dan ayat (2) dilaksanakan sesuai dengan peraturan perundang-undangan yang berlaku."
        },
        {
            "question": "Berapa lama waktu istirahat yang diberikan kepada pekerja/buruh?",
            "answer": "Pekerja/buruh berhak mendapatkan istirahat sekurang-kurangnya 1/2 jam setelah bekerja selama 4 jam terus menerus dan waktu istirahat tersebut tidak termasuk jam kerja.",
            "context": "Pasal 79: (1) Pengusaha wajib memberikan waktu istirahat dan cuti kepada pekerja/buruh. (2) Waktu istirahat dan cuti sebagaimana dimaksud dalam ayat (1), meliputi: a. istirahat antara jam kerja, sekurang-kurangnya setengah jam setelah bekerja selama 4 (empat) jam terus menerus dan waktu istirahat tersebut tidak termasuk jam kerja; b. istirahat mingguan 1 (satu) hari untuk 6 (enam) hari kerja dalam 1 (satu) minggu atau 2 (dua) hari untuk 5 (lima) hari kerja dalam 1 (satu) minggu; c. cuti tahunan, sekurang-kurangnya 12 (dua belas) hari kerja setelah pekerja/buruh yang bersangkutan bekerja selama 12 (dua belas) bulan secara terus menerus."
        }
    ]

def main():
    """
    Main function to run RAGAS evaluation
    """
    # Initialize components
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    # Create RAG chain
    llm = ChatOpenAI(temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    prompt = ChatPromptTemplate.from_template("""
    Kamu adalah asisten hukum yang ahli tentang regulasi ketenagakerjaan di Indonesia.
    Berdasarkan konteks berikut, jawablah pertanyaan dengan relevan, akurat, dan jelas.
    Sertakan nomor pasal dan ayat jika tersedia dalam konteks.
    
    KONTEKS:
    {context}
    
    PERTANYAAN:
    {question}
    """)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Get test questions
    test_questions = create_test_questions()
    
    # Run evaluation
    print("Running RAGAS evaluation...")
    results = evaluate_rag_chain(chain, vectorstore, test_questions)
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 40)
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    main() 