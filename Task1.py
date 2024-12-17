import pdfplumber
import fitz  
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

st.title("Task-1: Chat With PDF")
st.sidebar.title("PDF Scraper")
st.set_option('client.showErrorDetails', False)
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
process_pdf_clicked = st.sidebar.button("Process PDFs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = ChatGroq(
    temperature=0, 
    groq_api_key="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",
    model_name="llama-3.1-70b-versatile"
)

def extract_table_data(pdf_path):
    table_data = ""
    st.set_option('client.showErrorDetails', False)
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                for row in table:
                    if row:
                        row = [str(cell) for cell in row]
                        table_data += " | ".join(row) + "\n"
    return table_data

def extract_images_and_text(uploaded_file):
    image_data = ""
    try:
        if not uploaded_file:
            st.error("The uploaded file is empty. Please upload a valid PDF.")
            return image_data
        
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
                    image_data += pytesseract.image_to_string(img_pil) + "\n"
    except Exception as e:
        pass
    
    return image_data

if process_pdf_clicked:
    st.sidebar.success("Text Extracted Successfully")
    all_text = ""

    for uploaded_file in uploaded_files:
        extracted_text = extract_text(uploaded_file)
        all_text += extracted_text + "\n"

        table_data = extract_table_data(uploaded_file)
        all_text += table_data + "\n"

        image_data = extract_images_and_text(uploaded_file)
        all_text += image_data + "\n"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_text(all_text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore_openai = FAISS.from_texts(text_chunks, embeddings)

    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Ask a Question:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQA.from_llm(llm=llm, retriever=vectorstore.as_retriever())

        retriever = vectorstore.as_retriever()
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        result = chain.run(query)

        st.header("Answer")
        st.write(result)
