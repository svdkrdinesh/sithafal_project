import pdfplumber
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text
import os
import streamlit as st
import pickle
import time
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from PIL import Image
import pytesseract
import io

# Streamlit UI
st.title("Task-1: Chat With PDF")
st.sidebar.title("PDF Scraper")
st.set_option('client.showErrorDetails', False)
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
process_pdf_clicked = st.sidebar.button("Process PDFs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = ChatGroq(
    temperature=0, 
    groq_api_key="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",  # Replace with your actual Groq API key
    model_name="llama-3.1-70b-versatile"
)

# Function to extract text from tables using pdfplumber
def extract_table_data(pdf_path):
    table_data = ""
    st.set_option('client.showErrorDetails', False)
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                for row in table:
                    if row:  # Check if row is not empty
                        # Convert non-string elements in row to string
                        row = [str(cell) for cell in row]
                        table_data += " | ".join(row) + "\n"
    return table_data


# Function to extract images (graphs/pie charts) and perform OCR on them
def extract_images_and_text(uploaded_file):
    image_data = ""
    try:
        # Check if the uploaded file is empty
        if not uploaded_file:
            st.error("The uploaded file is empty. Please upload a valid PDF.")
            return image_data
        
        # Read the uploaded file into a bytes buffer
        pdf_bytes = uploaded_file.read()
        if len(pdf_bytes) == 0:
            st.error("The uploaded PDF is empty. Please upload a valid file.")
            return image_data

        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            for page_num in range(pdf.page_count):
                page = pdf[page_num]
                images = page.get_images(full=True)
                for img_index, img in enumerate(images):
                    xref = img[0]
                    base_image = pdf.extract_image(xref)
                    image_bytes = base_image["image"]
                    img_pil = Image.open(io.BytesIO(image_bytes))
                    # Use OCR to extract text from the image
                    image_data += pytesseract.image_to_string(img_pil) + "\n"
    except Exception as e:
        pass
    
    return image_data

if process_pdf_clicked:
    st.sidebar.success("Text Extracted Successfully")
    all_text = ""

    # Extract text from all PDFs
    for uploaded_file in uploaded_files:
        # Extract plain text
        extracted_text = extract_text(uploaded_file)
        all_text += extracted_text + "\n"

        # Extract table data
        table_data = extract_table_data(uploaded_file)
        all_text += table_data + "\n"

        # Extract images and perform OCR on graphs/pie charts
        image_data = extract_images_and_text(uploaded_file)
        all_text += image_data + "\n"

    # Split text into chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_text(all_text)

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore_openai = FAISS.from_texts(text_chunks, embeddings)

    # Save FAISS index to pickle
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file for future use
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

# Query input section
query = main_placeholder.text_input("Ask a Question:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQA.from_llm(llm=llm, retriever=vectorstore.as_retriever())

        # Retrieval-based chain
        retriever = vectorstore.as_retriever()
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # Get the answer for the query
        result = chain.run(query)

        # Display the answer
        st.header("Answer")
        st.write(result)
