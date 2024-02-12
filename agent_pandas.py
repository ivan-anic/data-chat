import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI

from langchain_experimental.agents import create_pandas_dataframe_agent


def chat_csv():
    """
    Implements the feature to chat with a LLM-based AI
    agent which has knowledge about the .csv file you
    manually provide.
    """
    # upload
    uploaded_file = st.sidebar.file_uploader("Upload File", type="csv")

    # clear
    if "messages" not in st.session_state or st.sidebar.button(
            "Clear conversation history"):
        st.session_state["messages"] = [{
            "role":
            "assistant",
            "content":
            "How can I help you?"
        }]

    # history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # prompt
    if prompt := st.chat_input(
            placeholder="What is this data about?"):
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        st.chat_message("user").write(prompt)

        if not uploaded_file:            
            st.session_state.messages.append({
                    "role": "assistant",
                    "content": "An error occured!"
                })
            st.chat_message("assistant").write("No csv uploaded!")
            return
        
        df = pd.read_csv(uploaded_file)

        llm = ChatOpenAI(temperature=0,
                         model="gpt-3.5-turbo-0613",
                         streaming=True)

        pandas_df_agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
        )

        with st.chat_message("assistant") as assistant:
            st_cb = StreamlitCallbackHandler(
                st.container(), expand_new_thoughts=False)
            try:
                response = pandas_df_agent.run(
                    st.session_state.messages, callbacks=[st_cb])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                st.write(response)
            except Exception as ex:
                print(ex)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "An error occured!"
                })
                st.write("An error occured!")
