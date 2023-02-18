import os
import enum
from pathlib import Path
from typing import List, Tuple, Any, Union

import torch
import pandas as pd
from dotenv import load_dotenv
import pinecone
from pinecone.core.client.model.query_response import QueryResponse
from sentence_transformers import SentenceTransformer

from embeddings import get_single_embedding
from config import ST_EMBEDDING_MODEL
from encode import Verse

load_dotenv()
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))


class EmbeddingType(enum.Enum):
    Ada = "oai_embeddings"
    SentenceTransfomer = "st_embeddings"

class Engine:
    def _get_query_vec(self, query: str, emb_type: EmbeddingType) -> torch.Tensor:
        if emb_type == EmbeddingType.Ada:
            return torch.tensor(get_single_embedding(query))
        elif emb_type == EmbeddingType.SentenceTransfomer:
            return self.model.encode(query, convert_to_tensor=True)
        else:
            raise Exception(f"No such embedding: {emb_type}")
    
    def search(
        self, 
        query: str,
        emb_type: EmbeddingType = EmbeddingType.Ada,
        only_text: bool = False
    ) -> List[Verse]:
        raise NotImplementedError

class SearchEngine(Engine):
    """Supports SentenceTransformer encoder model and OpenAI's Ada v2 embeddings."""

    def __init__(self, path: Path):
        self.df = pd.read_parquet(path, engine="fastparquet")
        self.model = SentenceTransformer(ST_EMBEDDING_MODEL)
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _get_query_vec(self, query: str, emb_type: EmbeddingType):
        return super()._get_query_vec(query, emb_type).to(self._device)

    def _get_embeddings(self, emb_type: EmbeddingType):
        return torch.tensor(self.df[emb_type.value]).to(self._device)

    def _get_search_results(
        self, query_vec: torch.Tensor, embeddings: torch.Tensor, source: pd.DataFrame, k: int = 10, only_text=False
    ) -> List[Verse]:
        """
        Cosine similarity: Ada embeddings are L2 normalized, so only require a dot product
        between the query and embedding vectors.
        """
        results = torch.topk(torch.matmul(embeddings, query_vec), k)
        results = source.loc[results.indices.cpu()][
            ["text"] if only_text else ["book", "chapter", "verse", "text"]
        ]
        return [
            Verse(None, None, None, np_row[0]) if only_text else Verse(*np_row)
            for np_row in list(results.values)
        ]

    def search(
        self,
        query: str,
        emb_type: EmbeddingType = EmbeddingType.Ada,
        only_text: bool = False,
    ) -> List[Verse]:
        query_vec = self._get_query_vec(query, emb_type)
        _embeddings = self._get_embeddings(emb_type)

        return self._get_search_results(
            query_vec, _embeddings, source=self.df, only_text=only_text
        )

class PineconeSearchEngine(Engine):
    def __init__(self, path: Path, index: Union[str, List]) -> None:

        self.df = pd.read_csv(path)
        self.model = SentenceTransformer(ST_EMBEDDING_MODEL)
        self._get_index(index)

    def _get_query_vec(self, query: str, emb_type: EmbeddingType) -> List[float]:
        return super()._get_query_vec(query, emb_type).tolist()

    def _get_index(self, index: Union[str, List]):
        index = [index] if isinstance(index, str) else index
        self.indices = {n: pinecone.Index(n) for n in index}
    
    def _get_search_results(
        self, 
        query_vec: List[float], 
        emb_type: EmbeddingType, 
        translation: str,
        only_text: bool,
        k: int = 10,
    ) -> List[Verse]:
        index = self.indices['ada' if emb_type == EmbeddingType.Ada else "mpnet"]
        results = index.query(query_vec, namespace=translation, top_k=k)
        return self._convert(results, only_text)

    def _convert(self, results: QueryResponse, only_text: bool) -> List[Verse]:
        """Maps Pinecone results to the book, chapter and verse"""
        if not results or not results.matches: 
            return []

        res = self.df.iloc[[int(r['id']) for r in results.matches]][
            ["text"] if only_text else ["book", "chapter", "verse", "text"]
        ]
        return [
            Verse(None, None, None, np_row[0]) if only_text else Verse(*np_row)
            for np_row in res.itertuples(index=None, name=None)
        ]

    def search(
        self,
        query: str,
        emb_type: EmbeddingType = EmbeddingType.Ada,
        only_text: bool = False,
        translation: str = "NKJV",
    ) -> List[Verse]:
        query_vec = self._get_query_vec(query, emb_type)
        return self._get_search_results(query_vec, emb_type, translation, only_text)

        
        