#!/bin/sh

set -eux

PYTHON="${PWD}/.venv/bin/python"

pinecone_embeddings() {
    "${PYTHON}" ./vector_db.py --namespace $1 --parquet_path ./data/$1_clean.parquet
}

pinecone_embeddings NKJV
pinecone_embeddings NIV