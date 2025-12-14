import streamlit as st

# Page Configuration
st.set_page_config(
    layout="centered",
    page_title="About - CPF Special Account Closure Chatbot"
)

# Header
st.title("About This Chatbot")

# Overview
st.markdown("""
This chatbot provides information about the **closure of the CPF Special Account at Age 55**, 
answering questions based on official government sources.
""")

st.divider()

# Project Scope Section
st.header("Project Scope")

with st.expander("📋 Objectives", expanded=True):
    st.markdown("""
    This chatbot answers questions using articles from government websites. For demonstration purposes, 
    it has been trained on specific CPF policy documents.
    
    **Safety Features:**
    - Guards against prompt injection by detecting malicious prompts
    - Validates that questions are relevant to the ingested policy documents
    - Uses multi-agent architecture for quality control
    """)

with st.expander("🗂️ Data Sources"):
    st.markdown("**Official articles from CPF and government websites (saved as PDFs):**")
    
    sources = [
        ("When can I start my retirement payouts?", "https://www.gov.sg/article/when-can-I-start-my-retirement-payouts"),
        ("Changes to CPF in 2024 and beyond", "https://www.cpf.gov.sg/member/infohub/educational-resources/changes-to-cpf-in-2024-and-beyond"),
        ("Closure of Special Account for members aged 55 and above", "https://www.cpf.gov.sg/member/infohub/educational-resources/closure-of-special-account-for-members-aged-55-and-above-in-early-2025"),
        ("What is the CPF Retirement Sum?", "https://www.cpf.gov.sg/member/infohub/educational-resources/what-is-the-cpf-retirement-sum"),
        ("Multiplying your CPF savings with compound interest", "https://www.cpf.gov.sg/member/infohub/educational-resources/multiplying-your-cpf-savings-with-compound-interest"),
        ("How to top up your CPF and the benefits", "https://www.cpf.gov.sg/member/infohub/educational-resources/how-to-top-up-your-cpf-and-the-benefits-of-doing-so"),
        ("Matching Grant for seniors who top up", "https://www.cpf.gov.sg/member/growing-your-savings/saving-more-with-cpf/matching-grant-for-seniors-who-top-up")
    ]
    
    for title, url in sources:
        st.markdown(f"- [{title}]({url})")

with st.expander("🔧 Technical Architecture"):
    st.markdown("""
    **Multi-Agent Workflow:**
    1. **Relevance Filtering** - Validates questions are relevant to CPF policy documents
    2. **Query Refinement** - Optimizes questions for better retrieval
    3. **Document Retrieval** - Searches documents using vector embeddings (RAG)
    4. **Response Generation** - Generates accurate answers with source citations
    
    **Tech Stack:**  
    Python | LangChain | CrewAI | OpenAI API | ChromaDB | Streamlit
    """)

st.divider()

# How to Use Section
st.header("How to Use This App")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 1️⃣ Enter Question")
    st.write("Type your question about CPF Special Account closure in the text area")

with col2:
    st.markdown("### 2️⃣ Submit")
    st.write("Click the 'Submit' button to process your question")

with col3:
    st.markdown("### 3️⃣ Get Answer")
    st.write("Receive an answer based on official CPF policy documents")

st.divider()

# Limitations
st.header("⚠️ Limitations & Disclaimer")

st.warning("""
**Important Notes:**
- This is a **demonstration project** for learning purposes
- Responses are limited to the scope of ingested articles
- Should **not be used** for actual financial planning or decision-making
- For official guidance, please refer to the [CPF website](https://www.cpf.gov.sg) or consult CPF directly
""")

st.divider()

# Footer
st.caption("Built with LangChain, CrewAI, and Streamlit | Personal Learning Project")