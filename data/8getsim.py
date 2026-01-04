import numpy as np
import pickle
import torch

# 加载 trajectory_features.pkl 文件
with open("/data/CaiZhuaoXiao/dataset/GB/trajectory_features.pkl", "rb") as f:
    traj_features = pickle.load(f)

# 获取 traj_ids 和特征数组
traj_ids = list(traj_features.keys())
features = np.array(list(traj_features.values()))

# 将特征数组转换为 PyTorch 张量，并移动到 GPU
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
features_tensor = torch.tensor(features, dtype=torch.float32, device=device)

def compute_similarity(sim_metric, user_emb):
    if sim_metric == "sin":
        score_matrix = torch.matmul(user_emb, user_emb.T)
    elif sim_metric == "cos":
        norm = user_emb.norm(dim=1, keepdim=True)
        user_emb = user_emb / norm
        score_matrix = torch.matmul(user_emb, user_emb.T)
    else:
        raise ValueError("Unsupported similarity metric")
    return score_matrix

def find_top_k_similar_traj(sim_metric, features_tensor, traj_ids, k=5):
    score_matrix = compute_similarity(sim_metric, features_tensor)
    score_matrix = score_matrix.cpu()
    
    top_k_traj = {}
    for i, traj_id in enumerate(traj_ids):
        similarity_scores = score_matrix[i]
        similarity_scores[i] = float('-inf')  # 排除自身相似度
        similar_indices = torch.argsort(similarity_scores, descending=True)[:k]
        top_k_traj[traj_id] = [traj_ids[idx] for idx in similar_indices.numpy()]
    
    return top_k_traj

# 示例：使用余弦相似度计算最相似的 20 个轨迹 ID
top_k_similar_traj = find_top_k_similar_traj("cos", features_tensor, traj_ids, k=20)

# 打印前 5 个结果
print("🔍 前 5 个 trajectory 的最相似轨迹（基于余弦相似度）:")
for i, (tid, similar_list) in enumerate(top_k_similar_traj.items()):
    print(f"{i+1}. 轨迹 {tid} 最相似的轨迹: {similar_list}")
    if i >= 4:
        break

# 保存为 pkl 文件
with open("/data/CaiZhuaoXiao/dataset/GB/top_k_similar_traj.pkl", "wb") as f:
    pickle.dump(top_k_similar_traj, f)

print("✅ 最相似的轨迹 ID 已保存为 top_k_similar_traj.pkl")

