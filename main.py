import re
import numpy as np
import pandas as pd
import pickle

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


app = FastAPI()

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

print("Loading model and dataset...")

word2vec_model = Word2Vec.load("word2vec_model.pkl")
df_master = pd.read_pickle("df_master.pkl")

print("Loaded successfully.")

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return words


def get_sentence_embedding(words):
    vectors = [
        word2vec_model.wv[word]
        for word in words
        if word in word2vec_model.wv
    ]

    if len(vectors) == 0:
        return np.zeros(word2vec_model.vector_size)

    return np.mean(vectors, axis=0)


print("Generating product embeddings...")

combined_texts = (
    df_master["title"].fillna("") + " " +
    df_master["description"].fillna("")
)

product_embeddings = []

for text in combined_texts:
    words = preprocess_text(text)
    emb = get_sentence_embedding(words)
    product_embeddings.append(emb)

product_embeddings = np.array(product_embeddings)

print("Embeddings ready.")


def recommend_products(query, n=10):

    query_words = preprocess_text(query)
    query_vector = get_sentence_embedding(query_words).reshape(1, -1)

    similarities = cosine_similarity(query_vector, product_embeddings)[0]

    top_indices = similarities.argsort()[::-1][:n]

    results = df_master.iloc[top_indices].copy()
    results["similarity_score"] = similarities[top_indices]

    return results[
        [
            "product_id",
            "title",
            "brand",
            "price",
            "rating",
            "platform",
            "image_url",
            "product_url",
            "similarity_score",
        ]
    ].to_dict(orient="records")



class QueryRequest(BaseModel):
    query: str
    n: int = 10


@app.post("/recommend")
def recommend(req: QueryRequest):
    return {
        "results": recommend_products(req.query, req.n)
    }