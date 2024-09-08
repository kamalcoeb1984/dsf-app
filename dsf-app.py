import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings 
from langchain.vectorstores import FAISS
import os

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def get_pdf_data(pdf_data):
  pdf_txt = ""
  for pdfdt in pdf_data:
    pdf_rdr = PdfReader(pdfdt)
    for pg in pdf_rdr.pages:
      pdf_txt += pg.extract_text()
  return pdf_txt

def get_txt_chunks(raw_data):
  
  txt_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
  )
  chunks_txt = txt_splitter.split_text(raw_data)
  return chunks_txt

def get_vectorstore(txt_chunks):
  # embeddings = OpenAIEmbeddings(openai_api_key = os.environ["OPENAI_API_KEY"])
  # embeddings = OpenAIEmbeddings()
  embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
  vector_store = FAISS.from_texts(texts = txt_chunks, embedding = embeddings)
  return vector_store


def main():
  st.write("Biits Project - Dailogue System Framework!")
  st.header("LLM using personal PDF documents :books:")
  st.text_input("Ask your queries:")

  with st.sidebar:
    load_dotenv()
    st.subheader("Internal Documents!!!")
    pdf_data = st.file_uploader("Upload PDFs and click 'Process'", accept_multiple_files = True)
    
    if st.button("Process"):

      with st.spinner("File processing"):
        #get pdf text
        raw_data = get_pdf_data(pdf_data)
        
        #get text chunks
        txt_chunks = get_txt_chunks(raw_data)
        
        #create vector store
        vStore = get_vectorstore(txt_chunks)


if __name__ == "__main__":
  main()
