import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA


st.set_page_config(page_title="Ask the text", page_icon="📖", layout="wide")
st.title("Ask the text")

txt = st.file_uploader("Upload a text file", type=["txt"])
openai_api_key = st.text_input("OpenAI API key", type="password")
if not openai_api_key.startswith("sk-"):
    st.warning("Please enter a valid OpenAI API key")
    st.stop()
if txt is not None:
    txt = txt.read().decode("utf-8")
    st.write(txt)
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = splitter.create_documents(txt)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    llm = OpenAI(openai_api_key=openai_api_key)
    qa = RetrievalQA.from_chain_type(llm, chain_type='stuff', retriever=retriever)
    question = st.text_input("Ask a question")
    if question:
        answer = qa.run(question)
        st.write(answer)