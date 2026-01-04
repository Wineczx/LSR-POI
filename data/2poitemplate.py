import json

# 文件路径
cate_file = "/data/CaiZhuaoXiao/dataset/GB/meta-GB.json"
review_file = "/data/CaiZhuaoXiao/dataset/GB/review_summary.json"
image_file = "/data/CaiZhuaoXiao/dataset/GB/image_description.json"
output_file = "generated_prompts.json"
def load_jsonlines_with_strip_prefix(filename):
    """
    读取每行 JSON 对象，key 是 'prefix_gmap_id'，将 key 的前缀去掉，只保留 gmap_id。
    返回 dict: {gmap_id: value}
    """
    data = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            for full_key, val in obj.items():
                # full_key 示例: "16_4d9a2f78d97ba1430b43336b"
                parts = full_key.split("_", 1)
                if len(parts) == 2:
                    gmap_id = parts[1]
                else:
                    gmap_id = full_key
                data[gmap_id] = val
    return data

def load_jsonlines_key_value_list(filename):
    """
    读取每行 JSON 对象，key 是 gmap_id，value 是列表。
    返回 dict: {gmap_id: list_of_texts}
    """
    data = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            for k, v in obj.items():
                data[k] = v
    return data

def load_cate(filename):
    data = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            gmap_id = obj.get("gmap_id")
            if gmap_id:
                data[gmap_id] = obj
    return data
def numbered_list_to_str(items, max_items=3):
    """
    把列表转换成 1. xxx. 2. xxx. ... 格式的一行字符串
    """
    parts = []
    for i, text in enumerate(items[:max_items], 1):
        # 确保句末有句号
        t = text.strip()
        if not t.endswith('.'):
            t += '.'
        parts.append(f"{i}. {t}")
    return " ".join(parts)
cate_data = load_cate(cate_file)
review_data = load_jsonlines_with_strip_prefix(review_file)
image_data = load_jsonlines_key_value_list(image_file)
with open(output_file, "w", encoding="utf-8") as out:
    for gmap_id, cate_obj in cate_data.items():
        category = cate_obj.get("category", "Unknown")

        reviews = review_data.get(gmap_id, [])
        images = image_data.get(gmap_id, [])

        # 跳过都空的
        if category == "null" and not reviews and not images:
            continue

        parts = [
            "You are given information about a Point of Interest (POI), including its category"
        ]
        parts.append(f"- Category: {category}.")

        if images:
            image_text = numbered_list_to_str(images)
            parts.append(f"- Image Description: {image_text}")

        if reviews:
            review_text = numbered_list_to_str(reviews)
            parts.append(f"- User Reviews: {review_text}")

        parts.append(
            "Generate one concise descriptive phrase (not a full sentence) that captures the key features, "
            "atmosphere, and function of the POI using meaningful keywords. The phrase should be around 5 to 10 words "
            "and follow a natural structure like 'a [adjective] [location or function] with [notable detail]'. "
            "Only output the phrase."
        )

        template = " ".join(parts)

        json_line = json.dumps({"id": gmap_id, "template": template}, ensure_ascii=False)
        out.write(json_line + "\n")