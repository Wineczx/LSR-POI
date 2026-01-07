import json

# 读取 meta.json（每行一个JSON对象）
meta_file = '/dataset/GB/meta-GB.json'
meta_dict = {}

with open(meta_file, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        gmap_id = item.get("gmap_id")
        if gmap_id:
            meta_dict[gmap_id] = item

# 处理 output.jsonl
output_file = '/dataset/GB/output.jsonl'
new_output_file = '/dataset/GB/poitemplate.jsonl'

with open(output_file, 'r', encoding='utf-8') as infile, open(new_output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        data = json.loads(line)
        gmap_id = data["id"]
        result = data.get("result", "None")

        if gmap_id in meta_dict:
            meta = meta_dict[gmap_id]
            category = meta.get("category", "Unknown")
            lat = meta.get("latitude", "Unknown")
            lon = meta.get("longitude", "Unknown")
        else:
            # 跳过 output.jsonl 中不在 meta.json 中的项
            continue

        # 构造新的 template
        new_template = (
            f"The point of interest has the following attributes: "
            f"category is {category}; located at latitude {lat} and longitude {lon}. "
            f"Description for the POI mention: {result}"
        )

        new_obj = {
            "id": gmap_id,
            "template": new_template
        }

        outfile.write(json.dumps(new_obj, ensure_ascii=False) + '\n')
