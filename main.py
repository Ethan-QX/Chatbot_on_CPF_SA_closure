import sqlite3
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from crewai import Agent, Task, Crew
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load vectordb and utilities
import Articles.load_articles
from Articles.load_articles import vectordb, llm, get_completion
from Articles.load_articles import security_advisor, relevance_checker
from Articles.load_articles import prompt_injection, check_relevance

# Create custom prompt template with your actual template
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer, and always provide the link to the article cited.
{context}
Question: {question}
Helpful Answer: if the question is not relevant, consider responding with "Wondering how the CPF Special Account closure impacts you? Ask me anything, and I'll help clarify the details!"
"""

prompt = ChatPromptTemplate.from_template(template)

# Create retriever
retriever = vectordb.as_retriever()

# Helper function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# New custom class that mimics old RetrievalQA behavior
class RAGChainWithSources:
    def __init__(self, retriever, llm, prompt):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt
    
    def invoke(self, question):
        docs = self.retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        formatted_prompt = self.prompt.format(context=context, question=question)
        answer = self.llm.invoke(formatted_prompt).content
        
        return {
            'result': answer,
            'source_documents': docs
        }

# Use the custom chain
qa_chain_with_sources = RAGChainWithSources(
    retriever=retriever,
    llm=ChatOpenAI(model='gpt-4o-mini'),
    prompt=prompt
)

# Streamlit App Configuration
st.set_page_config(
    layout="centered",
    page_title="Understanding the Closure of CPF Special Account"
)

st.title("Understanding the Closure of CPF Special Account")
st.markdown("**Try asking:**")
st.markdown("- How does the Special Account closure affect me?")
st.markdown("- When will my Special Account close?")
st.markdown("- What happens to my money after closure?")


form = st.form(key="form")
form.subheader("Prompt")

user_prompt = form.text_area("Enter your prompt here", height=200)


if form.form_submit_button("Submit"):
    
    st.toast(f"User Input Submitted - {user_prompt}")
    
    st.divider()
    
    # Get response with source documents
    response = qa_chain_with_sources.invoke(user_prompt)

    # Initialize an empty list for the document contents
    documents_content = []

    # Loop through the documents to collect their page contents
    for document in response['source_documents']:
        documents_content.append(document.page_content)

    inputs = {"user_prompt": user_prompt, "documents": documents_content}
    
    clarify = "Wondering how the CPF Special Account closure impacts you? Ask me anything, and I'll help clarify the details!"

    # Execute Task with Security Advisor First
    crew = Crew(
        agents=[security_advisor],
        tasks=[prompt_injection],
        verbose=True,
    )   
    malicious_check_result = crew.kickoff(inputs=inputs)

    # Only check relevance if prompt is not malicious
    if str(malicious_check_result) == "0":
        crew = Crew(
            agents=[relevance_checker],
            tasks=[check_relevance],
            verbose=True,
        )
        relevance_check_result = crew.kickoff(inputs=inputs)
        print(relevance_check_result)
        
        if str(relevance_check_result) == "1":
            answer = response['result']
        else: 
            answer = response['result']
    else:
        answer = response['result']
    with st.spinner("Analyzing your question..."):
        response = qa_chain_with_sources.invoke(user_prompt)
    st.write(answer)
    st.divider()
  
