import enum
import streamlit as st

from config import BOOKS
from search import PineconeSearchEngine, EmbeddingType

DATA = {
    "NKJV": "data/NKJV_fixed.csv",
    "NIV": "data/NIV_fixed.csv"
}

class Translation(enum.Enum):
    NIV = "NIV"
    NKJV = "NKJV"

@st.cache_resource
def load_engine():
    # for Pinecone, only `Ada` is available as its free tier only supports
    # a single index. Feel free to change this to multi-engine like in app.py
    # if you have access to multiple pinecone indices.
    return PineconeSearchEngine(DATA, index="ada")

engine = load_engine()

st.title("Search the Bible semantically:")

translation = st.radio("**Translations:**", [trans.value for trans in Translation])
emb_space = st.radio("**Embedding Space:**", [emb.name for emb in EmbeddingType if emb == EmbeddingType.Ada])

query = st.text_input("**Search:**", placeholder="What is the meaning of life?")


if query:
    with st.spinner(f'Searching verses in {translation} with {emb_space}...'):
        top_results = engine.search(query, translation=translation, emb_type=EmbeddingType[emb_space])
    
    if top_results:
        st.subheader("Results:")

    for result in top_results:
        with st.expander(f"{BOOKS[result.book]} {result.chapter}:{result.verse}", expanded=True):
            st.write(f"**{result.text}**")
        




