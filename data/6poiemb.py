import requests
import json
import pickle
from tqdm import tqdm
import os
import time

API_KEY = 'sk-976zO2MAnA6GcDR39417iJSA4cMiBBKkQtoSvHNCHXGjbTfU'
API_URL = "https://xiaoai.plus/v1/embeddings"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

input_file = "/data/CaiZhuaoXiao/dataset/GB/poitemplate.jsonl"
output_pkl = "/data/CaiZhuaoXiao/dataset/GB/poi_features.pkl"

BATCH_SIZE = 60
SAVE_EVERY = 100

# === 加载已有结果（容错） ===
def load_pickle_safe(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"⚠️ 警告：无法加载已保存的结果：{e}")
        return {}

# === 保存结果（容错） ===
def save_pickle_safe(data, path):
    try:
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"❌ 保存出错：{e}")

# === 批量获取嵌入，带重试机制 ===
def get_embeddings_batch(texts, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=HEADERS, json={
                "model": "text-embedding-3-small",
                "input": texts
            }, timeout=30)

            if response.status_code == 200:
                result = response.json()
                return [item["embedding"] for item in result["data"]]
            else:
                raise Exception(f"API 返回错误 {response.status_code}: {response.text}")
        except Exception as e:
            wait = 1.5 ** attempt
            print(f"⚠️ 第 {attempt+1} 次重试（等待 {wait:.1f}s）: {e}")
            time.sleep(wait)
    print("❌ 最终失败，跳过该 batch")
    return [None] * len(texts)

# === 主处理逻辑 ===
def main():
    poi_embeddings = load_pickle_safe(output_pkl)

    with open(input_file, "r", encoding="utf-8") as f:
        all_lines = [json.loads(line) for line in f if line.strip()]

    # 去掉已处理的
    to_process = [item for item in all_lines if item["id"] not in poi_embeddings]
    print(f"🚀 需处理 POIs 数量: {len(to_process)}")

    for i in tqdm(range(0, len(to_process), BATCH_SIZE), desc="Embedding Batches"):
        batch = to_process[i:i + BATCH_SIZE]
        texts = [item["template"] for item in batch]
        ids = [item["id"] for item in batch]

        embeddings = get_embeddings_batch(texts)

        for poi_id, emb in zip(ids, embeddings):
            if emb is not None:
                poi_embeddings[poi_id] = emb
            else:
                print(f"跳过 {poi_id}（嵌入失败）")

        # 保存中间结果
        if (i + BATCH_SIZE) % SAVE_EVERY < BATCH_SIZE:
            save_pickle_safe(poi_embeddings, output_pkl)
            print(f"💾 中间保存完成（已嵌入 {len(poi_embeddings)} 个 POI）")

        time.sleep(1.0)  # 防止频繁请求

    # 最终保存
    save_pickle_safe(poi_embeddings, output_pkl)
    print(f"\n✅ 全部完成，总共嵌入：{len(poi_embeddings)} 个 POI")

if __name__ == "__main__":
    main()
