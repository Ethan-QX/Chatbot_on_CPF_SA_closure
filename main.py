# Set up and run this Streamlit App
import streamlit as st
from crewai import Agent, Task, Crew
import pandas as pd
from langchain_openai import ChatOpenAI
# from helper_functions import llm
from langchain.chains import RetrievalQA
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
#load in vectordb, use rag to answer prompt
import Articles.load_articles
from Articles.load_articles import vectordb
from Articles.load_articles import llm
from langchain.prompts import PromptTemplate
from Articles.load_articles import get_completion


from Articles.load_articles import security_advisor, relevance_checker

from Articles.load_articles import prompt_injection, check_relevance

#simple RAG
rag_chain=RetrievalQA.from_llm(
    retriever=vectordb.as_retriever(), llm=llm,return_source_documents=True)


# # with a custom prompt
# template =  """Use the following pieces of context to answer the question at the end.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
# {context}
# Question: {question}
# Helpful Answer:"""
# QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
# # Run chain
# qa_chain = RetrievalQA.from_chain_type(
#     ChatOpenAI(model='gpt-4o-mini'),
#     retriever=vectordb.as_retriever(),
#     return_source_documents=True, # Make inspection of document possible
#     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
# )

# print(qa_chain_multiquery.invoke(rewritten))


# llm_response = rag_chain.invoke('how does it affect me i am 25?')
# print(llm_response['result'])

from helper_functions.utility import check_password  

# Check if the password is correct.  
if not check_password():  
    st.stop()


# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="Understanding the Closure of CPF Special Account"
)
# endregion <--------- Streamlit App Configuration --------->

st.title("Understanding the Closure of CPF Special Account")

form = st.form(key="form")
form.subheader("Prompt")

user_prompt = form.text_area("Enter your prompt here", height=200)

if form.form_submit_button("Submit"):
    
    st.toast(f"User Input Submitted - {user_prompt}")
    
    st.divider()
    
    response=rag_chain.invoke(user_prompt)
    # #test execution

    # Initialize an empty list for the document contents
    documents_content = []

    # Loop through the documents to collect their page contents
    for document in response['source_documents']:
        documents_content.append(document.page_content)

    inputs = {"user_prompt": user_prompt, "documents": documents_content}
    
    clarify="I didn't quite catch that. Could you please rephrase your question for me?"

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
            answer=response['result']

        else: 
            answer=clarify


    else:
        answer=clarify
        
        #retrieve source documents


        # response, course_details = process_user_message(user_prompt)
    st.write(answer)

    st.divider()

 
