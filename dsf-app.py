import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
import os
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

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
  embeddings = OpenAIEmbeddings(model="text-embedding-3-large",
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024
                               )
  # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
  vectorstore = FAISS.from_texts(texts=txt_chunks, embedding=embeddings)
  return vectorstore

def get_conversation_chain(vStore):
    #llm = ChatOpenAI()
    llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
 )
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vStore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
  try:
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
    
    
    st.header("Dailogue System!!!")
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.write("LLM using personal PDF documents :books:")
    user_question = st.text_input("Ask your queries:")
    if user_question:
        handle_userinput(user_question)
      
    with st.sidebar:
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

          # create conversation chain
          st.session_state.conversation = get_conversation_chain(vStore)
  except Exception as e: 
    st.write('Oops somthing is wrong. Please see below for issue: \n', e)
if __name__ == "__main__":
  main()
