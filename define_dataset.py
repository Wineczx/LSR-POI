from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
class TrajectoryDatasetTrain(Dataset):
    def __init__(self, train_df, poi_id2idx_dict, user_id2idx_dict, args):
        self.df = train_df
        self.args = args
        self.num_sim = args.num_sim
        self.poi_id2idx_dict = poi_id2idx_dict  # 将 poi_id2idx_dict 保存到实例中
        self.similar_traj_dict = self.load_similar_traj(args.data_sim)  # 加载相似轨迹文件
        self.traj_seqs = []  # traj id: user id + traj no.
        self.input_seqs = []
        self.label_seqs = []

        # 遍历所有轨迹
        for traj_id in tqdm(sorted(set(train_df['trajectory_id'].tolist()))):
            traj_df = train_df[train_df['trajectory_id'] == traj_id]
            poi_ids = traj_df['POI_id'].to_list()
            poi_idxs = [poi_id2idx_dict[each] for each in poi_ids]
            time_feature = traj_df[args.time_feature].to_list()

            input_seq = []
            label_seq = []
            for i in range(len(poi_idxs) - 1):
                input_seq.append((poi_idxs[i], time_feature[i]))
                label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))

            if len(input_seq) < args.short_traj_thres:
                continue

            self.traj_seqs.append(traj_id)
            self.input_seqs.append(input_seq)
            self.label_seqs.append(label_seq)

    def load_similar_traj(self, similar_traj_file):
        with open(similar_traj_file, 'rb') as f:  # 以二进制模式打开
            similar_traj_dict = pickle.load(f)
        return similar_traj_dict

    def get_full_traj(self, traj_id):
        """
        获取指定轨迹的完整信息（整条轨迹的POI和时间特征），并将 POI 映射到索引。
        """
        traj_df = self.df[self.df['trajectory_id'] == traj_id]
        poi_ids = traj_df['POI_id'].to_list()
        time_feature = traj_df[self.args.time_feature].to_list()

        # 使用 poi_id2idx_dict 映射 POI
        mapped_traj = [(self.poi_id2idx_dict.get(poi_id, -1), time) for poi_id, time in zip(poi_ids, time_feature)]
        return mapped_traj  # 返回映射后的轨迹信息

    def __len__(self):
        return len(self.traj_seqs)

    def __getitem__(self, index):
        """
        返回当前轨迹及其相似轨迹的信息（包括 POI 映射后的索引）。
        """
        traj_id = self.traj_seqs[index]
        input_seq = self.input_seqs[index]
        label_seq = self.label_seqs[index]

        # 获取相似轨迹的信息（整条轨迹）
        similar_traj_ids = self.similar_traj_dict.get(traj_id, [])[:self.num_sim]  # 限制为最多num_sim条
        similar_trajs = []
        for similar_id in similar_traj_ids:
            if similar_id not in self.df['trajectory_id'].values:
                continue
            
            # 获取完整轨迹，并使用 poi_id2idx_dict 映射 POI
            mapped_full_traj = self.get_full_traj(similar_id)

            similar_trajs.append({
                "traj_id": similar_id,
                "full_traj": mapped_full_traj  # 将 POI 映射后的轨迹信息
            })

        return [traj_id, input_seq, label_seq, similar_trajs]

# class TrajectoryDatasetTrain(Dataset):
#     def __init__(self, train_df,poi_id2idx_dict,user_id2idx_dict,args):
#         self.df = train_df
#         self.traj_seqs = []  # traj id: user id + traj no.
#         self.input_seqs = []
#         self.label_seqs = []

#         for traj_id in tqdm(sorted(set(train_df['trajectory_id'].tolist()))):
#             traj_df = train_df[train_df['trajectory_id'] == traj_id]
#             poi_ids = traj_df['POI_id'].to_list()
#             poi_idxs = [poi_id2idx_dict[each] for each in poi_ids]
#             time_feature = traj_df[args.time_feature].to_list()

#             input_seq = []
#             label_seq = []
#             for i in range(len(poi_idxs) - 1):
#                 input_seq.append((poi_idxs[i], time_feature[i]))
#                 label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))

#             if len(input_seq) < args.short_traj_thres:
#                 continue

#             self.traj_seqs.append(traj_id)
#             self.input_seqs.append(input_seq)
#             self.label_seqs.append(label_seq)

#     def __len__(self):
#         assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
#         return len(self.traj_seqs)

#     def __getitem__(self, index):
#         return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

class TrajectoryDatasetVal(Dataset):
    def __init__(self, df,poi_id2idx_dict,user_id2idx_dict,args):
        self.df = df
        self.traj_seqs = []
        self.input_seqs = []
        self.label_seqs = []

        for traj_id in tqdm(sorted(set(df['trajectory_id'].tolist()))):
            user_id = traj_id.split('_')[0]

            # Ignore user if not in training set
            if user_id not in user_id2idx_dict.keys():
                continue

            # Ger POIs idx in this trajectory
            traj_df = df[df['trajectory_id'] == traj_id]
            poi_ids = traj_df['POI_id'].to_list()
            poi_idxs = []
            time_feature = traj_df[args.time_feature].to_list()

            for each in poi_ids:
                if each in poi_id2idx_dict.keys():
                    poi_idxs.append(poi_id2idx_dict[each])
                else:
                    # Ignore poi if not in training set
                    continue

            # Construct input seq and label seq
            input_seq = []
            label_seq = []
            for i in range(len(poi_idxs) - 1):
                input_seq.append((poi_idxs[i], time_feature[i]))
                label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))

            # Ignore seq if too short
            if len(input_seq) < args.short_traj_thres:
                continue

            self.input_seqs.append(input_seq)
            self.label_seqs.append(label_seq)
            self.traj_seqs.append(traj_id)

    def __len__(self):
        assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
        return len(self.traj_seqs)

    def __getitem__(self, index):
        return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])