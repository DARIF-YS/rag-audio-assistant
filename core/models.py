import os
from dotenv import load_dotenv
import whisper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

# Load .env variables
load_dotenv()

@st.cache_resource(show_spinner=False)
def _load_models_cached():
    whisper_model = whisper.load_model("small")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = Chroma(
        collection_name="audio_transcripts",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    os.environ["GOOGLE_API_KEY"] =  os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    return whisper_model, vector_store, llm


def load_models():
    """Wrapper to load models with a Streamlit spinner."""
    import streamlit as st
    with st.spinner("Loading models..."):
        return _load_models_cached()
