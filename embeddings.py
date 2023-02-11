import os
import openai
from typing import List
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-ada-002"


def get_single_embedding(text: str, model=EMBEDDING_MODEL) -> List[float]:
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


def get_multi_embeddings(texts: List[str], model=EMBEDDING_MODEL) -> List[List[float]]:
    texts = [text.replace("\n", " ") for text in texts]
    return [
        data["embedding"]
        for data in openai.Embedding.create(input=texts, model=model)["data"]
    ]
