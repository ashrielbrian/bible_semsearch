import enum
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class Verse:
    book: Optional[int]
    chapter: Optional[int]
    verse: Optional[int]
    text: str


@dataclass
class VerseEmbeddings(Verse):
    oai_embeddings: List[float]
    st_embeddings: List[float]

class EmbeddingType(enum.Enum):
    Ada = "oai_embeddings"
    SentenceTransfomer = "st_embeddings"