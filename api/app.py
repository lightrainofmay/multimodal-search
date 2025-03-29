from flask import Flask, request, jsonify
from flask_cors import CORS
from app.embedder import load_and_embed
from app.search import extract_keywords, semantic_search, process_results
import os

app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your_key_here")

# 加载模型和索引
df, index, model = load_and_embed(
    json_path="data/jino_all_media.json",
    embedding_path="data/text_embeddings.pkl",
    index_path="data/faiss_index.bin"
)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("message", "").strip()
    if not query:
        return jsonify({"error": "Message cannot be empty"}), 400

    keyword = extract_keywords(query, OPENAI_API_KEY)
    text_results = semantic_search(keyword, df, index, model)
    media = process_results(df, text_results)

    return jsonify({"query": query, "search_results": media})
