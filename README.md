
# Introduction

Semantic search of the Bible, comparing OpenAI's Ada v2 and `SentenceTransfomer` embeddings.

*Inspired by Christoffer Rehn's* [`bible-search`](https://github.com/hoffa/bible-search).


# Getting Started

### 1. Setting up dependencies:
```bash
    make init
```

### 2. Generate embeddings:
```bash
    make embeddings
```

**Important**: You must have an `OPENAI_API_KEY` in a `.env` file in the project directory when running this command.

`make embeddings` does two things:
1. Generates OpenAI [`text-embedding-ada-002`](https://openai.com/blog/new-and-improved-embedding-model/) embeddings using the OpenAI API.
2. Generates `SentenceTransformer` embeddings using the `ST_EMBEDDING_MODEL` defined in `config.py`. You can change this to any encoder model from their [docs](https://www.sbert.net/docs/pretrained_models.html). Default: `all-mpnet-base-v2`

# Usage
After `make embeddings`,

```python
    from search import SearchEngine, EmbeddingType

    nkjv_engine = SearchEngine(path="data/NKJV_clean.parquet")
    top_verses = nkjv_engine.search(
        "what is the meaning of life?", 
        emb_type=EmbeddingType.SentenceTransfomer, 
        only_text=True
    )
```

```bash
    # top_verses output
    [
        ('For what has man for all his labor and for the striving of his heart with which he has toiled under the sun?',),
        ('to cast out all your enemies from before you as the LORD has spoken.',),
        ('to know the love of Christ which passes knowledge; that you may be filled with all the fullness of God.',),
        ('to do whatever Your hand and Your purpose determined before to be done.',),
        ('For to me to live is Christ and to die is gain.',),
        ('and to make all see what is the fellowship of the mystery which from the beginning of the ages has been hidden in God who created all things through Jesus Christ;',),
        ('knowing that from the Lord you will receive the reward of the inheritance; for you serve the Lord Christ.',),
        ('To do justice to the fatherless and the oppressed That the man of the earth may oppress no more.',),
        ('That Your way may be known on earth Your salvation among all nations.',),
        ('being filled with the fruits of righteousness which are by Jesus Christ to the glory and praise of God.',)
    ]
```


# Data
- Bible versions (NIV, NKJV) are sourced from [my-bible-study](http://my-bible-study.appspot.com)

*Soli Deo Gloria.*