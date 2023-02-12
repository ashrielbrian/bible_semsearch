import os
import csv
from pathlib import Path
from typing import Iterator, List, Any
from dataclasses import dataclass

import pandas as pd

from embeddings import (
    get_single_embedding,
    get_multi_embeddings,
    get_st_transformer_embeddings,
)
from config import get_st_model

model = get_st_model()


@dataclass
class Row:
    book: int
    chapter: int
    verse: int
    text: str


@dataclass
class RowEmbeddings(Row):
    oai_embeddings: List[float]
    st_embeddings: List[float]


def _get_row(fp: Path) -> Iterator[Row]:
    with fp.open() as f:
        for row in csv.DictReader(f):
            yield Row(
                int(row["book"]), int(row["chapter"]), int(row["verse"]), row["text"]
            )


def generate_embeddings(fp: Path):
    for row in _get_row(fp):
        yield RowEmbeddings(
            row.book,
            row.chapter,
            row.verse,
            row.text,
            get_single_embedding(row.text),
            get_st_transformer_embeddings([row.text], model)[0],
        )


def _read_row_batches(fp: Path, batch_size: int = 50) -> Iterator[List[Row]]:
    with fp.open() as f:
        batches = []
        for row in csv.DictReader(f):
            batches.append(
                Row(
                    int(row["book"]),
                    int(row["chapter"]),
                    int(row["verse"]),
                    row["text"],
                )
            )
            if len(batches) == batch_size:
                yield batches
                batches = []

        if batches:
            yield batches


def generate_embeddings_batch(
    fp: Path, batch_size: int = 50
) -> Iterator[List[RowEmbeddings]]:
    for batch in _read_row_batches(fp, batch_size):
        try:
            batch_text = [row.text for row in batch]
            ada_embs = get_multi_embeddings(batch_text)
            st_embs = get_st_transformer_embeddings(batch_text, model)
            yield [
                RowEmbeddings(
                    row.book, row.chapter, row.verse, row.text, ada_emb, st_emb
                )
                for ada_emb, st_emb, row in zip(ada_embs, st_embs, batch)
            ]
        except Exception as e:
            import traceback

            print(traceback.format_exc())
            print(f"Problematic batch: {batch}")
            raise e


def append_to_parquet(fp: Path, data: Any):
    df = pd.DataFrame(
        data,
        columns=["book", "chapter", "verse", "text", "oai_embeddings", "st_embeddings"],
    )
    df.to_parquet(
        fp,
        compression="gzip",
        engine="fastparquet",
        index=False,
        append=os.path.isfile(fp),
    )


def write_to_parquet(csv_fp: Path, parquet_fp: Path):
    for batch in generate_embeddings_batch(csv_fp):
        append_to_parquet(
            parquet_fp,
            batch,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv_path",
        type=Path,
        required=True,
        help="Path to the source CSV Bible translation (book,chapter,verse,text) columns.",
    )

    parser.add_argument(
        "--parquet_path",
        type=Path,
        required=True,
        help="Destination path to the output parquet file containing the embeddings \
            (book,chapter,verse,text,oai_embeddings,st_embeddings) column.",
    )

    args = parser.parse_args()

    write_to_parquet(args.csv_path, args.parquet_path)


if __name__ == "__main__":
    main()
