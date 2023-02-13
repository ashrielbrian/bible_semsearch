from pathlib import Path
import pandas as pd

def clean(path: Path):
    df = pd.read_csv(
        path, 
        sep=',', 
        escapechar='\\', 
        names=['book', 'chapter', 'verse', 'text']
    )
    df.dropna(inplace=True)
    df.to_csv(path, index=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv_path",
        type=Path,
        required=True,
        help="Path to the source CSV Bible translation (book,chapter,verse,text) columns.",
    )

    args = parser.parse_args()

    clean(args.csv_path)