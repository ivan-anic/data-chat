from dotenv import load_dotenv
from agent_pdf import chat_pdf
from agent_pandas import chat_csv
import streamlit as st


def intro():
    st.write("# Welcome! 👋")
    st.sidebar.success("Pick an agent")
      

def init_streamlit():
    st.set_page_config(page_title="Data Chat")
    st.header("Chat with your data")
    page_names_to_funcs = {
        "—": intro,
        "Chat with .pdf": chat_pdf,
        "Chat with .csv": chat_csv,
    }

    sidebar = st.sidebar.selectbox("Choose an agent", page_names_to_funcs.keys())
    page_names_to_funcs[sidebar]()


def main():
    load_dotenv()
    init_streamlit()


if __name__ == "__main__":
    main()
