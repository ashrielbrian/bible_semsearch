from sentence_transformers import SentenceTransformer

ST_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"


def get_st_model(model: str = ST_EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model)
