#!/bin/sh

set -eux

PYTHON="${PWD}/.venv/bin/python"

generate_embeddings() {
    "${PYTHON}" ./clean.py --csv_path ./data/$1_fixed.csv
    "${PYTHON}" ./encode.py --csv_path ./data/$1_fixed.csv --parquet_path ./data/$1_clean.parquet
}

generate_embeddings NKJV
generate_embeddings NIV