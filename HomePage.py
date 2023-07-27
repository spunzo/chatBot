import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space


st.set_page_config(layout="wide")

# Customize the sidebar
markdown = """
GitHub Repository: <https://github.com/spunzo/chatBot>
"""

st.sidebar.title("About")
st.sidebar.info(markdown)
logo = "https://cdn.dribbble.com/userupload/6610160/file/original-99c5681554021552faca5c4e3c24e65e.png?resize=400x0"
st.sidebar.image(logo)
st.sidebar.write("Powered by Simone Punzo")


# Customize page title
st.title("Streamlit for Langchain Conversational Agents with Langchain 🦜")

st.markdown(
    """
    This multipage app template demonstrates two interactive webapps created using [Streamlit](https://streamlit.io) and [Langchain](<https://python.langchain.com/docs/get_started/introduction.html/>).\n\n
    The two pages correspond to two conversation agents fine-tuned on Italian Laws regulating Division of Assets after Divorce and Inheritance.\n\n
    The chatbots differ only for the chain used.
    """
)

st.header("Click on a page from the sidebar at the top and let's start playing!")


