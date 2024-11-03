import streamlit as st

# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="My Streamlit App"
)
# endregion <--------- Streamlit App Configuration --------->

st.title("About this App")

st.write("This is a chatbot that provides users with information relating to the closure of the Special Account at Age 55.")

"Project Scope:"
"objectives: This chatbot will answer questions on articles taken from government websites, for demostration purposes it will "
"data sources: Articles available on CPF's website, and Factually, saved as PDFs"
"https://www.gov.sg/article/when-can-I-start-my-retirement-payouts"
"https://www.cpf.gov.sg/member/infohub/educational-resources/changes-to-cpf-in-2024-and-beyond"
"https://www.cpf.gov.sg/member/infohub/educational-resources/closure-of-special-account-for-members-aged-55-and-above-in-early-2025"
"https://www.cpf.gov.sg/member/infohub/educational-resources/what-is-the-cpf-retirement-sum"
"https://www.cpf.gov.sg/member/infohub/educational-resources/multiplying-your-cpf-savings-with-compound-interest"
"https://www.cpf.gov.sg/member/infohub/educational-resources/how-to-top-up-your-cpf-and-the-benefits-of-doing-so"
"https://www.cpf.gov.sg/member/growing-your-savings/saving-more-with-cpf/matching-grant-for-seniors-who-top-up"

"features:"



with st.expander("How to use this App"):
    st.write("1. Enter your prompt in the text area.")
    st.write("2. Click the 'Submit' button.")
    st.write("3. The app will generate a text completion based on your prompt.")
