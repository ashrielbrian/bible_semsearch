import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pinecone
import numpy as np
import pandas as pd
from dotenv import load_dotenv


load_dotenv()
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))

INDEX = {
    # each index requires a new pinecone pod. free tier only allows a single pod.
    # "mpnet": "st_embeddings"

    "ada": "oai_embeddings",
}

def batcher(df: pd.DataFrame, batch_size: int = 300) -> Iterator[pd.DataFrame]:
    splits = round(len(df) / batch_size)

    if splits <= 1:
        yield df
    else:
        for chunk in np.array_split(df, splits):
            yield chunk


def create_index(index_name: str, dims: int, delete_if_exists = True):
    if delete_if_exists and index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)
    
    if not index_name in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=dims, pod_type="p2.x1", metric="dotproduct")
    
    return pinecone.Index(index_name=index_name)


def batch_upsert(index: pinecone.Index, df: pd.DataFrame, column: str, namespace: str):
    for batch_df in batcher(df):
        index.upsert(vectors=zip(batch_df.index.values.astype(str), batch_df[column]), namespace=namespace)


def build(name: str, fp: Path, delete_if_exists: bool):
    df = pd.read_parquet(fp, engine="fastparquet")

    indices = {k: create_index(k, len(df[v][0]), delete_if_exists) for k, v in INDEX.items()}

    for key, index in indices.items():
        batch_upsert(index, df, column=INDEX[key], namespace=name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--namespace', type=str, help="Translation name, e.g. `NIV`.")
    parser.add_argument('--parquet_path', type=Path, help="Path/URL to the parquet file containing the vector embeddings.")
    parser.add_argument('--new_index', action="store_true", help="Whether to delete and create a new index if an existing one exists.")

    args = parser.parse_args()
    
    build(name=args.namespace, fp=args.parquet_path, delete_if_exists=args.new_index)