import enum
import streamlit as st

from config import BOOKS
from search import PineconeSearchEngine, SearchEngine, EmbeddingType

PINECONE_DEPLOYMENT = True

DATA = {
    "NKJV": "data/NKJV_fixed.csv",
    "NIV": "data/NIV_fixed.csv"
} if PINECONE_DEPLOYMENT else {
    "NKJV": "data/NKJV_clean.parquet", # "https://www.dropbox.com/s/wd3kxh012jfhjya/NKJV_clean.parquet?dl=1",
    "NIV": "data/NIV_clean.parquet" # "https://www.dropbox.com/s/78jm8wh4cqhvwwv/NIV_clean.parquet?dl=1",
}

class Translation(enum.Enum):
    NIV = "NIV"
    NKJV = "NKJV"

@st.cache_resource
def load_pinecone_engine():
    # for Pinecone, only `Ada` is available as its free tier only supports
    # a single index. Feel free to change this to multi-engine like in app.py
    # if you have access to multiple pinecone indices.
    return PineconeSearchEngine(DATA, index="ada")

@st.cache_resource
def load_engine():
    # search using in-memory parquet files
    return SearchEngine(DATA)

st.title("Search the Bible semantically:")

if PINECONE_DEPLOYMENT:
    st.caption("Using Pinecone.")

translation = st.radio("**Translations:**", [trans.value for trans in Translation])
emb_space = st.radio("**Embedding Space:**", [
    emb.name for emb in EmbeddingType if emb == EmbeddingType.Ada
] if PINECONE_DEPLOYMENT else [emb.name for emb in EmbeddingType])

query = st.text_input("**Search:**", placeholder="What is the meaning of life?")


if query:
    engine = load_pinecone_engine() if PINECONE_DEPLOYMENT else load_engine()

    with st.spinner(f'Searching verses in {translation} with {emb_space}...'):
        top_results = engine.search(query, translation=translation, emb_type=EmbeddingType[emb_space])
    
    if top_results:
        st.subheader("Results:")

    for result in top_results:
        with st.expander(f"{BOOKS[result.book]} {result.chapter}:{result.verse}", expanded=True):
            st.write(f"**{result.text}**")
        

st.caption("You can find the codebase at this Github [repo](https://github.com/ashrielbrian/bible_semsearch).")


