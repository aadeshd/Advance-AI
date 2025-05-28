from backend import run_llm
import streamlit as st
from typing import Set
st.header("Documentation Helper Bot")

prompt = st.text_input("Prompt", placeholder="Enter your question here ..")

if (
    "user_prompt_history" not in st.session_state
    and "chat_answer_history" not in st.session_state
    and "chat_history" not in st.session_state):
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_answer_history"] = []
    st.session_state["chat_history"] = []

def create_src_str(urls: Set[str]) -> str:
    if not urls:
        return ""
    src_lst = list(urls)
    src_lst.sort()
    src_string = "Sources: \n"
    for i, source in enumerate(urls):
        src_string += f"{i+1}. {source} \n"
    return src_string


if prompt:
    with st.spinner("Generating response"):

        response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])
        sources = set([doc.metadata['source']for doc in response['context']])

        formatted_response = (
            f"{response['answer']} \n\n {create_src_str(sources)}"
        )

        st.session_state['user_prompt_history'].append(prompt)
        st.session_state['chat_answer_history'].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", response['answer']))

if st.session_state['chat_answer_history']:
    for response, query in zip(st.session_state['chat_answer_history'], st.session_state['user_prompt_history']):
        st.chat_message("user").write(query)
        st.chat_message("assistant").write(response)