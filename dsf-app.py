import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import pdfReader

def get_pdf_data(pdf_data):
  pdf_txt = ""
  for pdf in pdf_data:
    pdf_rdr = pdfReader(pdf)
    for pg in pdf_rdr.pages:
      pdf_txt += pg.extract_text()
  return pdf_txt


def main():
  st.write("Biits Project - Dailogue System Framework!")
  st.header("LLM using personal PDF documents :books:")
  st.text_input("Ask your queries:")

  with st.sidebar:
    st.subheader("Internal Documents :pdf:")
    pdf_data = st.file_uploader("Upload PDFs and click 'Process'", accept_multiple_files = True)
    
    if st.button("Process"):

      with st.spinner("File processing"):
        #get pdf text
        raw_data = get_pdf_data(pdf_data)
        st.write(raw_data)
        #get text chunks

        #create vector store

if __name__ == "__main__":
  main()
