import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings 
from langchain.vectorstores import FAISS
import os
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

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
  embeddings = OpenAIEmbeddings()
  # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
  vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
  return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
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
  load_dotenv()
  st.write("Biits Project - Dailogue System Framework!")
  
  st.header("LLM using personal PDF documents :books:")
  st.text_input("Ask your queries:")
  st.write(css, unsafe_allow_html=True)

  if "conversation" not in st.session_state:
      st.session_state.conversation = None
  if "chat_history" not in st.session_state:
      st.session_state.chat_history = None

  st.header("Chat with multiple PDFs :books:")
  user_question = st.text_input("Ask a question about your documents:")
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
        st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
  main()
