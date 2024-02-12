from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
import streamlit as st


@st.cache_resource
def load_chain(file):
    """
    Loads the LLM question answering chain with 
    the provided file
    
    Args:
        file : the pdf file
    
    Returns:
        document_search, chain : document search, qa chain
    """
    pdfreader = PdfReader(file)
    raw_text = ''
    for _, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

    # split under token size
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 800,
        chunk_overlap  = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    # chain
    embeddings = OpenAIEmbeddings()
    document_search = FAISS.from_texts(texts, embeddings)
    llm=ChatOpenAI(
        temperature=0.6,
        model="gpt-3.5-turbo-16k"
    )
    chain = load_qa_chain(llm, chain_type="stuff")

    return document_search, chain


def chat_pdf():
    """
    Implements the feature to chat with a LLM-based AI
    agent which has knowledge about the .pdf file you
    manually provide.
    """
    # upload
    uploaded_file = st.sidebar.file_uploader("Upload File", type="pdf")

    if uploaded_file:
        document_search, chain = load_chain(uploaded_file)

    input = st.text_input("Input:", key="input")
    submit = st.button("Ask the question")

    if submit:
        if not uploaded_file:
            st.error("No file uploaded!")
            return
        query = input
        docs = document_search.similarity_search(query)        
        st.subheader("The response is")
        resp = chain.run(input_documents=docs, question=query)
        st.write(resp.strip())
