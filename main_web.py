import argparse
import openai
from prompts import SYSTEM_PROMPT
import streamlit as st

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=False, type=str, default='gpt-3.5-turbo', help='openai model, default gpt-3.5-turbo')
parser.add_argument('--max_tokens', required=False, type=int, default=1024, help='max_tokens, default 1024')
parser.add_argument('--temperature', required=False, type=float, default=0, help='temperature, default 0')
parser.add_argument('--max_history_len', required=False, type=int, default=5, help='max history length, default 5')
args = parser.parse_args()

def main_web():
    st.title("Chat With ChatGPT")

    # openai.api_key = st.secrets["OPENAI_API_KEY"]

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = args.model

    if "messages" not in st.session_state:
        st.session_state['messages'] = []
        print(st.session_state.keys())
        st.session_state['messages'].append({"role": "system", "content": SYSTEM_PROMPT})

    for message in st.session_state['messages']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state['messages'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state['messages'][-args.max_history_len:]
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state['messages'].append({"role": "assistant", "content": full_response})

main_web()