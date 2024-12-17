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

###bash
pip install streamlit pdfplumber pymupdf pdfminer.six pillow pytesseract langchain langchain-groq faiss-cpu sentence-transformers
Streamlit run task1. py 

# Web Crawling, Embedding, and Response Generation System

This project crawls and scrapes text data from websites, generates embeddings for the scraped text, and then allows querying of the information using a pre-trained language model. The system uses the following components:

- **Web Scraping**: Scrapes content from given URLs.
- **Sentence Embeddings**: Embeds the scraped text using the `SentenceTransformer` model.
- **FAISS Index**: Stores and searches the text embeddings using FAISS for fast retrieval.
- **Query Handling**: Processes user queries by searching for the most relevant chunks of text.
- **Response Generation**: Generates answers to user queries using a pre-trained GPT model.

## Requirements

To run this project, you'll need the following dependencies:

- `requests`: For sending HTTP requests to websites.
- `beautifulsoup4`: For scraping text content from the websites.
- `sentence-transformers`: For generating sentence embeddings.
- `faiss-cpu` or `faiss-gpu`: For building and querying the embedding index.
- `scikit-learn`: For machine learning utilities.
- `transformers`: For using pre-trained models for natural language processing.
- `huggingface_hub`: For HuggingFace model integration.

You can install the required dependencies using the following:

```bash
pip install ipywidgets
Python task2.py 

