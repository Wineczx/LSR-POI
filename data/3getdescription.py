import json
from tqdm import tqdm
from transformers import pipeline
import torch
import os

# 输入输出文件路径
input_file = "generated_prompts.json"
output_file = "output.jsonl"
temp_file = "output.tmp.jsonl"

# 加载模型
model_id = "/data/CaiZhuaoXiao/Llama-3-8b"
pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16},
    device="cuda",
)

# 判断是否已有部分输出，支持断点续跑
processed_ids = set()
if os.path.exists(temp_file):
    with open(temp_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "id" in obj:
                processed_ids.add(obj["id"])

# 打开输出文件追加写
with open(input_file, "r", encoding="utf-8") as fin, \
     open(temp_file, "a", encoding="utf-8") as fout:

    for line in tqdm(fin, desc="Processing"):
        obj = json.loads(line)

        if obj["id"] in processed_ids:
            continue  # 跳过已处理

        # 生成 Prompt
        messages = [
            {"role": "system", "content": "You are a helpful human assistant."},
            {"role": "user", "content": obj["template"]}
        ]
        prompt = pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        terminators = [
            pipe.tokenizer.eos_token_id,
            pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # 生成结果
        output = pipe(
            prompt,
            max_new_tokens=64,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )
        full_text = output[0]["generated_text"]
        reply = full_text[len(prompt):].strip()

        # 添加结果字段并写入
        obj["result"] = reply
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fout.flush()  # 每条都写入磁盘

# 重命名为最终文件名
os.rename(temp_file, output_file)
print(f"完成处理，输出保存在 {output_file}")
