# Chat With PDF

This project allows users to upload PDF files, extract text, tables, and images, perform OCR on graphs/charts, and interact with the processed content using a conversational AI chatbot powered by **Groq LLM** and **FAISS** embeddings.

## Features

1. **PDF Upload and Processing**
   - Supports multiple PDF uploads.
   - Extracts plain text, table data, and images.
   - OCR is performed on images like graphs, pie charts, and other visuals.

2. **Text Splitting and Vectorization**
   - Uses `RecursiveCharacterTextSplitter` to split the extracted content into manageable chunks.
   - Creates embeddings using the **HuggingFace** model `sentence-transformers/all-MiniLM-L6-v2`.

3. **Conversational AI**
   - Powered by **Groq LLM** (`llama-3.1-70b-versatile`) for question-answering.
   - Retrieves answers based on vectorized content using **FAISS** (Facebook AI Similarity Search).

4. **Data Persistence**
   - Saves FAISS vector store to a pickle file (`faiss_store_openai.pkl`) for future use.

## Installation

### Prerequisites

Ensure you have the following tools installed:
- Python (>=3.8)
- pip
- Tesseract OCR (for image text extraction)

### Install Dependencies

Run the following command in your terminal:

```bash
pip install streamlit pdfplumber pymupdf pdfminer.six pillow pytesseract langchain langchain-groq faiss-cpu sentence-transformers
Streamlit run task1. py
