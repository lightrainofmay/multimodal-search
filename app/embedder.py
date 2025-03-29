import os
import json
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

def load_and_embed(json_path, embedding_path, index_path, model_name="moka-ai/m3e-base"):
    with open(json_path, "r", encoding="utf-8") as f:
        media_entries = json.load(f)

    df = pd.DataFrame(media_entries)
    df["text"] = df["text"].fillna("")
    df["enhanced_text"] = df["text"].apply(lambda x: f"{x} {x.lower()}")

    model = SentenceTransformer(model_name)

    if os.path.exists(index_path) and os.path.exists(embedding_path):
        print("✅ 加载已有索引")
        index = faiss.read_index(index_path)
        with open(embedding_path, "rb") as f:
            embeddings = pickle.load(f)
    else:
        print("🔧 正在计算嵌入")
        df["embedding"] = df["enhanced_text"].apply(
            lambda x: model.encode(x, normalize_embeddings=True) if x.strip() else np.zeros(768)
        )
        embeddings = np.vstack(df["embedding"].values).astype(np.float32)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        faiss.write_index(index, index_path)
        with open(embedding_path, "wb") as f:
            pickle.dump(embeddings, f)

    return df, index, model
