import enum
import streamlit as st

from config import BOOKS
from search import SearchEngine, EmbeddingType

DATA = {
    "NKJV": "https://www.dropbox.com/s/wd3kxh012jfhjya/NKJV_clean.parquet?dl=1",
    "NIV": "https://www.dropbox.com/s/78jm8wh4cqhvwwv/NIV_clean.parquet?dl=1",
}

class Translation(enum.Enum):
    # streamlit Cloud community only allows up to 1GB storage,
    # so we'll only be loading the NKJV version here.
    # Feel free to uncomment NIV below if you're running locally.
    # NIV = "NIV"
    NKJV = "NKJV"

@st.cache_resource
def load_engine(translation_type: str) -> SearchEngine:
    return SearchEngine(DATA[translation_type])

engines = {
    trans.value: load_engine(trans.value) for trans in Translation
}

st.write("Search the Bible (NIV, NKJV) by meaning.")

translation = st.radio("Translations:", [trans.value for trans in Translation])
emb_space = st.radio("Embedding Space:", [emb.name for emb in EmbeddingType])

query = st.text_input("Search:", placeholder="What is the meaning of life?")


if query:
    with st.spinner(f'Searching verses in {translation} with {emb_space}...'):
        top_results = engines[translation].search(query, emb_type=EmbeddingType[emb_space])
    
    for result in top_results:
        with st.expander(f"{BOOKS[result[0]]} {result[1]}:{result[2]}", expanded=True):
            st.write(f"**{result[3]}**")
        




