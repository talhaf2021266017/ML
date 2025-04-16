import streamlit as st
import requests
import json
from datetime import datetime
import threading
import os

OLLAMA_URL = "http://localhost:11434/api/chat"
LOG_FILE = "chat_history.jsonl"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]
if "last_input" not in st.session_state:
    st.session_state.last_input = ""

# Page configuration
st.set_page_config(page_title="DeepSeek Chatbot", page_icon="🤖")
st.title("💬 DeepSeek Chatbot")

# Custom CSS for styling
st.markdown("""
<style>
.thinking {
    color: #888;
    font-style: italic;
}
.final-response {
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)

# Log messages to JSONL
def log_message_pair(user_msg, assistant_msg):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user": user_msg,
        "assistant": assistant_msg
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# Simulated training function
def load_dataset(path=LOG_FILE):
    data = []
    try:
        with open(path, "r") as f:
            for line in f:
                data.append(json.loads(line))
    except FileNotFoundError:
        print("No chat history file found.")
    return data

def simulate_training(data):
    if not data:
        print("No data to train on.")
        return
    print(f"Loaded {len(data)} chat pairs for training.\n")
    for i, pair in enumerate(data):
        print(f"Training on pair {i + 1}:")
        print(f" 🧑 User: {pair['user']}")
        print(f" 🤖 Assistant: {pair['assistant']}\n")
    print("✅ Training simulation complete.")

# Sidebar
with st.sidebar:
    st.header("Chat History")
    if len(st.session_state.messages) > 1:
        chat_text = "\n\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages])
        st.download_button(
            label="Download Chat",
            data=chat_text,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

    if st.button("Clear History"):
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]
        st.session_state.last_input = ""
        st.rerun()

    if st.button("Start Background Training"):
        def background_training():
            data = load_dataset()
            simulate_training(data)
        threading.Thread(target=background_training).start()
        st.write("🚀 Training started in the background!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and model response
user_input = st.chat_input("Type your message...")

if user_input and user_input != st.session_state.last_input:
    st.session_state.last_input = user_input
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown(
            '<div class="thinking">DeepSeek R1 is thinking...</div>',
            unsafe_allow_html=True
        )

        response_placeholder = st.empty()
        full_response = ""

        payload = {
            "model": "deepseek-r1:1.5b",
            "messages": st.session_state.messages.copy(),
            "stream": True
        }

        try:
            with requests.post(OLLAMA_URL, json=payload, stream=True) as response:
                if response.status_code != 200:
                    raise Exception(f"API request failed with status code {response.status_code}")

                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode('utf-8'))
                        if 'message' in data and 'content' in data['message']:
                            token = data['message']['content']
                            full_response += token
                            response_placeholder.markdown(
                                f'<div class="final-response">{full_response}</div>',
                                unsafe_allow_html=True
                            )
                thinking_placeholder.empty()

        except Exception as e:
            thinking_placeholder.empty()
            full_response = f"Sorry, I encountered an error: {str(e)}"
            response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    log_message_pair(user_input, full_response)
