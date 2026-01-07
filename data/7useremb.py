
import json
import pickle
import os
import time
from tqdm import tqdm
from openai import OpenAI, OpenAIError

# 初始化 OpenAI 客户端
client = OpenAI(
    base_url='',
    api_key = "" 
)

# ======= 批量获取嵌入（带重试、超时）=======
def get_batch_response(prompts, max_retries=5, backoff_factor=1.5):
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=prompts,
            )
            return [item.embedding for item in response.data]
        except OpenAIError as e:
            wait_time = backoff_factor ** attempt
            print(f"⚠️ 第 {attempt + 1} 次重试，等待 {wait_time:.1f}s，错误：{e}")
            time.sleep(wait_time)
    print("❌ 最终重试失败，跳过该 batch")
    return [None] * len(prompts)

# ======= 安全加载 pickle =======
def load_pickle_safe(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"⚠️ 警告：加载 pickle 文件失败（可能损坏），错误：{e}")
        print("➡️ 将从空字典开始继续处理。")
        return {}

# ======= 主处理流程 =======
def process_templates(input_file, output_file, save_every=100, batch_size=10):
    # 加载已有特征
    trajectory_features = load_pickle_safe(output_file)
    print(f"✅ 已加载已有特征数量: {len(trajectory_features)}")

    # 读取所有轨迹数据
    with open(input_file, 'r', encoding='utf-8') as f:
        all_items = [json.loads(line.strip()) for line in f]

    # 过滤未处理的
    to_process = [item for item in all_items if item["trajectory_id"] not in trajectory_features]
    print(f"🚀 需要处理的新轨迹数: {len(to_process)}")

    # 批量处理
    for i in tqdm(range(0, len(to_process), batch_size), desc="Embedding Batches"):
        batch = to_process[i:i + batch_size]
        prompts = [item["template"] for item in batch]
        traj_ids = [item["trajectory_id"] for item in batch]

        embeddings = get_batch_response(prompts)

        for traj_id, embedding in zip(traj_ids, embeddings):
            if embedding is not None:
                trajectory_features[traj_id] = embedding
            else:
                print(f"[x] 跳过 {traj_id}（请求失败）")

        # 保存中间结果
        if (i + batch_size) % save_every < batch_size:
            with open(output_file, 'wb') as out_f:
                pickle.dump(trajectory_features, out_f)
            print(f"💾 已保存 {len(trajectory_features)} 条嵌入")
        time.sleep(1)

    # 最后保存
    with open(output_file, 'wb') as out_f:
        pickle.dump(trajectory_features, out_f)

    print(f"\n✅ 全部完成！总共保存特征数：{len(trajectory_features)}")

# ======= 执行入口 =======
if __name__ == "__main__":
    input_file = ''
    output_file = ''

    process_templates(input_file, output_file, save_every=300, batch_size=40)
