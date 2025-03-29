import openai
import numpy as np

def extract_keywords(query, api_key):
    openai.api_key = api_key

    prompt = f"""请从下面的中文问题中提取最核心的搜索关键词：

1. 如果问题是 **"基诺语的X怎么说？"**，你应该只返回 **"X"**，不要返回 "基诺语"。
2. 仅返回一个最相关的关键词。
3. 不要返回句子或多余的解释，只返回关键词。

问题：{query}
关键词："""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5
    )
    return response.choices[0].message.content.strip()

def semantic_search(query, df, index, model, top_k=5):
    query_embedding = model.encode(query, normalize_embeddings=True).reshape(1, -1)
    _, indices = index.search(query_embedding, top_k)
    valid_indices = [i for i in indices[0] if i < len(df)]
    return df.iloc[valid_indices]["text"].tolist()

def process_results(df, text_results):
    file_results = df[["text", "file"]].dropna().drop_duplicates()
    text_to_files = file_results.groupby("text")["file"].apply(list).to_dict()

    output = {}
    for text in text_results:
        files = text_to_files.get(text, [])
        output[text] = {
            "images": [f for f in files if f.endswith((".jpg", ".png", ".webp"))] or ["暂无图片"],
            "audios": [f for f in files if f.endswith((".mp3", ".wav", ".ogg"))] or ["暂无音频"],
        }
    return output
