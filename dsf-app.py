import streamlit as st

def main():
    st.set_page_config(page_title = "Biits Project - Dailogue System Framework", page_icon = ":books:")
    
    st.header("LLM using personal PDF documents :books:")
    st.text_input("Ask your queries:")
    
    with st.sidebar:
        st.subheader("Internal Documents")
        st.file_uploader("Upload PDFs and click 'Process'")
        st.button("Process")
    
    
if __name__ == '__main()__':
    main()
