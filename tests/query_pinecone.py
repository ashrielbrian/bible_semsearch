import os, sys
from typing import List

# adds current dir to the PYTHONPATH 
sys.path.insert(0, ".")

from dotenv import load_dotenv
import pandas as pd
import pinecone
from pinecone.core.client.model.query_response import QueryResponse

from models import Verse

TRANSLATION = "NKJV"
EXPECTED_VERSE = 'In whose hand is the life of every living thing And the breath of all mankind?'

df = pd.read_csv(f"data/{TRANSLATION}_fixed.csv")

# load .env file when running locally
load_dotenv()
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))

def load_sample_vector(fp: str):
    with open(fp, 'r') as f:
        query_vec = eval(f.read())
    return query_vec

def get_search_results(
        query_vec: List[float], 
        translation: str,
        k: int = 10
    ) -> List[Verse]:
        index = pinecone.Index("ada")
        results = index.query(query_vec, namespace=translation, top_k=k)
        return _convert(results, translation)

def _convert(results: QueryResponse, translation: str) -> List[Verse]:
    """Maps Pinecone results to the book, chapter and verse"""
    if not results or not results.matches: 
        return []

    print(results.matches)
    res = df.iloc[[int(r['id']) for r in results.matches]][["book", "chapter", "verse", "text"]]
    return [Verse(*np_row) for np_row in res.itertuples(index=None, name=None)]

def query_index(query_vec: List):
    top_verses = get_search_results(
        query_vec, 
        TRANSLATION
    )
    assert top_verses[0].text == EXPECTED_VERSE
    return top_verses

if __name__ == "__main__":
    sample_vec = load_sample_vector("tests/sample_vector.txt")
    print(f"Sample vector has {len(sample_vec)} dims")
    print(query_index(sample_vec))