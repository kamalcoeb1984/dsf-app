import streamlit as st

def main():
  st.write("Biits Project - Dailogue System Framework!")
  st.header("LLM using personal PDF documents :books:")
  st.text_input("Ask your queries:")

  with st.sidebar:
    st.subheader("Internal Documents :pdf:")
    st.file_uploader("Upload PDFs and click 'Process'")
    st.button("Process")    

if __name__ == "__main__":
  main()
