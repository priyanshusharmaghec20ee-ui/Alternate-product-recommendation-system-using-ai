import re
import numpy as np
import pandas as pd
import pickle
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import uvicorn

app = FastAPI()

print("🚀 Loading models and data...")

# ---------------------------
# LOAD FILES
# ---------------------------

# Load Word2Vec (saved via pickle)
with open("word2vec_model.pkl", "rb") as f:
    word2vec_model = pickle.load(f)

# Load dataset
df_master = pd.read_pickle("df_master.pkl")

# Load FAISS index
faiss_index = faiss.read_index("faiss_index.bin")

# Load index mapping
with open("indices.pkl", "rb") as f:
    indices = pickle.load(f)

print("✅ All files loaded successfully.")

# ---------------------------
# NLP SETUP
# ---------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

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

# ---------------------------
# RECOMMENDATION FUNCTION
# ---------------------------

def recommend_products(query, n=10):

    query_words = preprocess_text(query)
    query_vector = get_sentence_embedding(query_words)

    query_vector = np.array([query_vector]).astype("float32")

    # FAISS search
    distances, faiss_indices = faiss_index.search(query_vector, n)

    matched_indices = [indices[i] for i in faiss_indices[0]]

    results = df_master.iloc[matched_indices].copy()

    results["similarity_score"] = distances[0]

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


# ---------------------------
# API STRUCTURE
# ---------------------------

class QueryRequest(BaseModel):
    query: str
    n: int = 10


@app.post("/recommend")
def recommend(req: QueryRequest):
    return {"results": recommend_products(req.query, req.n)}


# ---------------------------
# LOCAL RUN
# ---------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
