import os
from dotenv import load_dotenv
import streamlit as st
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings

# Initialize Ollama
llm = Ollama(model="llama2:3b")
embeddings = OllamaEmbeddings(model="llama2:3b")

def get_embedding(input):
    """Get embeddings using Ollama"""
    try:
        embedding = embeddings.embed_query(input)
        return [embedding]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def get_completion(prompt, temperature=0, max_tokens=1024):
    """Single prompt completion using Ollama"""
    try:
        response = llm.predict(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response
    except Exception as e:
        print(f"Error getting completion: {e}")
        return None

def get_completion_by_messages(messages, temperature=0, max_tokens=1024):
    """Handle chat-style messages using Ollama"""
    try:
        # Convert messages to a single prompt
        prompt = ""
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            prompt += f"{role}: {content}\n"
        
        response = llm.predict(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response
    except Exception as e:
        print(f"Error getting completion: {e}")
        return None

def count_tokens(text):
    """Estimate token count"""
    # Simple estimation - Llama tokenization is different but this gives rough estimate
    words = text.split()
    return len(words) * 1.3  # Rough estimate of tokens per word

def count_tokens_from_message(messages):
    """Estimate token count from messages"""
    total_text = ""
    for message in messages:
        content = message.get("content", "")
        total_text += content + " "
    return count_tokens(total_text)

# Example usage in Streamlit
def main():
    st.title("Llama 3B Chat Interface")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat input
    user_input = st.text_input("Your message:", key="user_input")

    if st.button("Send"):
        if user_input:
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Get response
            response = get_completion_by_messages(st.session_state.messages)
            
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.write(f"You: {content}")
        else:
            st.write(f"Assistant: {content}")

if __name__ == "__main__":
    main()
