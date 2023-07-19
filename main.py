import streamlit as st
from streamlit_chat import message
from streamlit_extras.add_vertical_space import add_vertical_space
import model
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os

load_dotenv()

topics = os.getenv('topics','default')

cohere_api_key = os.getenv('COHERE_API_KEY', 'YourAPIKeyIfNotSet')



def extract_source_documents(response : dict):
    source_documents = [elem.page_content for elem in response["source_documents"]]
    relevant_documents =[" ".join(elem.split()[:3]) for elem in source_documents]
    return relevant_documents

def extract_text(response):
    return response["text"]


st.set_page_config(page_title="Legal Assistant", page_icon=":robot:")
st.header("Legal Assistant")

with st.sidebar:
    st.title('Your personal legal assistant on Division of assets and Inheritance is here! :male-judge:')
    st.markdown("""
    LLM-powered Streamlit application.
    Framework:
            - [LANGCHAIN] : (<https://python.langchain.com/docs/get_started/introduction.html/>)
            - [STREAMLIT] :(<https://streamlit.io/>)
            - [CHROMADB] : (<https://www.trychroma.com/>)
            - [COHERE] : (<https://cohere.com/>)
    """)
    add_vertical_space(5)
    st.write("Powered by Simone Punzo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

placeholder = "Let's start! Write your question here!";

def get_text():
    input_text = st.text_input("You: ", placeholder, key="input")
    return input_text


user_input = get_text()

if placeholder not in user_input:
    output = model.run_chain(user_input,topics)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

