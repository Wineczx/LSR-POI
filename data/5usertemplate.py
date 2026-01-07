import pandas as pd
import json
from collections import defaultdict

# 读取 train.csv
train_df = pd.read_csv("/dataset/GB/GB_train.csv")

# 读取 GB_meta.json（每行一个 json 对象）
poi_meta = {}
with open("/dataset/GB/meta-GB.json", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        poi_id = data["gmap_id"]
        poi_meta[poi_id] = data

# Step 1: 收集所有 POI 并分配全局新编号
unique_pois = set(train_df["POI_id"])
poi_id_map = {poi: idx for idx, poi in enumerate(sorted(unique_pois))}  # 从 0 开始编号

# Step 2: 为每个用户整理访问的 POI
user_poi_dict = defaultdict(list)

for _, row in train_df.iterrows():
    trajectory_id = row["trajectory_id"]
    poi_id = row["POI_id"]
    if poi_id in poi_meta:
        new_poi_id = poi_id_map[poi_id]  # 映射为全局统一编号
        category = poi_meta[poi_id].get("category", "Unknown")
        lat = poi_meta[poi_id].get("latitude", "0.0")
        lon = poi_meta[poi_id].get("longitude", "0.0")
        user_poi_dict[trajectory_id].append((new_poi_id, category, lat, lon))

# Step 3: 生成模板并保存为 JSON 行格式
with open("/dataset/GB/user_templates.jsonl", "w", encoding="utf-8") as out_file:
    for trajectory_id, poi_list in user_poi_dict.items():
        poi_strs = []
        for idx, (new_poi_id, category, lat, lon) in enumerate(poi_list):
            poi_strs.append(f"{idx+1}. POI ID: {new_poi_id}, {category} ({lat}, {lon})")
        poi_str_joined = "; ".join(poi_strs)
        template = f"The user has visited the following points of interest: {poi_str_joined} Please conclude the user's preference."
        out_file.write(json.dumps({"trajectory_id": str(trajectory_id), "template": template}, ensure_ascii=False) + "\n")
