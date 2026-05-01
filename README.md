# Indonesian Labor Law Question Answering System

## Overview
This repository contains a final-year thesis (**Skripsi**) project implementing a Retrieval-Augmented Generation (RAG) system for Question Answering on Indonesian Labor Law. The system is designed to provide accurate, contextually relevant answers to complex legal questions, effectively mitigating the severe issue of "hallucination" commonly found in generative Large Language Models (LLMs).

### Academic Summary
Navigating Indonesian labor regulations is notoriously difficult due to complex clauses and scattered information. While LLMs offer a way to rapidly query information, they are prone to legal hallucinations—producing convincing but factually incorrect legal advice because they lack verifiable context. This thesis proposes a Retrieval-Augmented Generation (RAG) architecture that grounds an LLM (Qwen3 8B) in the actual text of Indonesian Labor Laws. By implementing a hybrid retrieval strategy consisting of TF-IDF and dense embedding similarity, harmonized via Reciprocal Rank Fusion (RRF), the system dramatically improved legal question-answering accuracy to 94.44% on expert-verified test questions.

## Problem Statement
Labor regulations in Indonesia play a crucial role in protecting workers' rights. However, the technical legal language and dispersion of rules across various documents make it difficult for ordinary workers to understand their rights. While Large Language Models (LLMs) can assist in answering legal queries, they frequently suffer from **hallucination**—generating confident but factually incorrect answers because they merely predict text without real-time access to the actual legal statutes. In the legal domain, such logic failures can lead to severe misinterpretation.

## Proposed Solution
To overcome LLM hallucination, this project utilizes a **Retrieval-Augmented Generation (RAG)** approach. Before the LLM generates a response, the system first retrieves the most relevant articles and clauses directly from the vector-indexed labor laws. This retrieved factual context is then injected into the prompt, forcing the LLM to generate answers strictly grounded in actual legislation.

## Features
- **Accurate Legal Q&A**: Answers are firmly grounded in actual legislations, stating explicit chapters and articles.
- **Hybrid Document Retrieval**: Combines keyword-based search with semantic vector search for highly relevant context gathering.
- **Context Transparency**: Displays the exact legal articles and matching scores (TF-IDF, Embeddings, and RRF) used.
- **Comparison Mode**: Native UI toggle to evaluate outputs between the ungrounded standard LLM (No RAG) and the RAG-enhanced pipeline.

## System Architecture
The system pipeline follows advanced RAG architecture:
1. **Retrieval**: User question is matched against the vector database and text documents using a Hybrid Retrieval strategy.
2. **Augmentation**: The retrieved contexts (relevant articles/clauses) are compiled and injected into the LLM's prompt via LangChain.
3. **Generation**: The LLM synthesizes the final response strictly based on the injected context, preventing any assumptions.

## Methodology
- **Data Source**: Indonesian Law No. 13 of 2003 on Manpower, properly updated with the amendments in Law No. 6 of 2023 Article 81 (Job Creation Law).
- **Preprocessing**: Data cleansing, case folding, exact article adjustment, and chunking performed to preserve legal context boundaries.
- **Embedding**: OpenAI's `text-embedding-3-small` used to convert text segments into dense contextual vectors.
- **Hybrid Retrieval**:
  - *Sparse Retrieval*: TF-IDF with Term Frequency/Inverse Document Frequency keyword matching.
  - *Dense Retrieval*: Cosine similarity using the ChromaDB vector database.
- **Reciprocal Rank Fusion (RRF)**: A specialized weighted algorithm to harmonize and re-rank the retrieved chunks from both dense (70% weight) and sparse (30% weight) retrievers.
- **Generative Model**: Qwen3 8B, chosen for its excellent balance of computational efficiency and robust linguistic performance.

## Technologies Used
- **Language**: Python
- **Frameworks**: LangChain, Streamlit
- **Large Language Model (LLM)**: Qwen3 8B (Served locally via Ollama)
- **Embedding Model**: OpenAI `text-embedding-3-small`
- **Vector Database**: ChromaDB
- **Machine Learning**: Scikit-Learn (TF-IDF and Cosine computations)

## Evaluation
The architecture was comprehensively evaluated using the **RAGAS (Retrieval-Augmented Generation Assessment)** framework:
- **Faithfulness (0.926)**: Measures how factually accurate the generated answer is compared strictly to the retrieved context.
- **Answer Relevancy (0.860)**: Evaluates how relevant the generated answer is to the user's initial query.
- **Context Precision (0.935)**: Assesses whether the systems ranks the ground-truth contexts highly.
- **Context Recall (0.949)**: Measures the system's ability to retrieve all the necessary context required to answer the query fully.

## Results
- The system was tested against 18 complex, multi-layered legal questions validated by labor law experts.
- Achieved **94.44% accuracy**, with 17 out of 18 answers strictly complying with the legal substance.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/putramkti/rag-hukum-ketenagakerjaan.git
   cd rag-hukum-ketenagakerjaan
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Environment Variables. Create a `.env` file in the root directory and add your OpenAI Key (used for embedding the queries):
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. Install and run [Ollama](https://ollama.com/), then pull the required Qwen model:
   ```bash
   ollama pull qwen3:8b
   ```

## Usage
Run the Streamlit web application:
```bash
streamlit run rag.py
```
This will open the user interface in your default web browser on `http://localhost:8501`. Enter queries and switch between "Dengan RAG" (With RAG) and "Tanpa RAG" (Without RAG) modes to witness the difference in accuracy.

## Project Structure
- `rag.py`: Main Streamlit application and core RAG pipeline implementation.
- `rag_proses_perhitungan.py`: Analytical scripts for text-chunk computations and metric extraction calculations.
- `notebook_fix.ipynb`: Jupyter Notebook detailing the data exploratory process and experimentation phase.
- `data/`: Contains the raw and processed text references of the Indonesian Labor Laws.
- `chroma_db_.../`: Persisted local vector store directory containing the embedded document chunks.

## Research Contribution
This project highlights the vast potential of applying AI within the Indonesian legal sector. By decisively solving the critical flaw of LLM hallucinations using a state-of-the-art Hybrid Retrieval architecture, this research contributes to democratizing legal knowledge. It demonstrates how modern systems can make hyper-complex legal regulations accessible, reliable, and easily understandable for ordinary workers and practitioners alike.

## Author
- **Putra Jadi Mukti**
- *Universitas Pembangunan Nasional “Veteran” Yogyakarta (2025)*
