import enum
from pathlib import Path
from typing import List, Tuple, Any

import torch
import pandas as pd

from sentence_transformers import SentenceTransformer
from embeddings import get_single_embedding

ST_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class EmbeddingType(enum.Enum):
    Ada = "ada"
    miniLM = "minilm"


class SearchEngine:
    """Supports SentenceTransformer encoder model and OpenAI's Ada v2 embeddings."""

    def __init__(self, path: Path):
        self.df = pd.read_parquet(path, engine="fastparquet")
        self.model = SentenceTransformer(ST_EMBEDDING_MODEL)

    def _get_query_vec(self, query: str, emb_type: EmbeddingType):
        if emb_type == EmbeddingType.Ada:
            return torch.tensor(get_single_embedding(query))
        elif emb_type == EmbeddingType.miniLM:
            return self.model.encode(query, convert_to_tensor=True)
        else:
            raise Exception(f"No such embedding: {emb_type}")

    def _get_embeddings(self, emb_type: EmbeddingType):
        if emb_type == EmbeddingType.Ada:
            return torch.tensor(self.df["oai_embeddings"])
        elif emb_type == EmbeddingType.miniLM:
            return torch.tensor(self.df["minilm_embeddings"])
        else:
            raise Exception(f"No such embedding: {emb_type}")

    def _get_search_results(
        self, query_vec, embeddings, source: pd.DataFrame, k: int = 10, only_text=False
    ) -> List[Tuple[Any]]:
        """
        Cosine similarity: Ada embeddings are L2 normalized, so only require a dot product
        between the query and embedding vectors.
        """
        results = torch.topk(torch.matmul(embeddings, query_vec), k)
        results = source.loc[results.indices][
            ["text"] if only_text else ["book", "chapter", "verse", "text"]
        ]
        return [tuple(np_row) for np_row in list(results.values)]

    def search(
        self,
        query: str,
        emb_type: EmbeddingType = EmbeddingType.Ada,
        only_text: bool = False,
    ):
        query_vec = self._get_query_vec(query, emb_type)
        _embeddings = self._get_embeddings(emb_type)

        return self._get_search_results(
            query_vec, _embeddings, source=self.df, only_text=only_text
        )