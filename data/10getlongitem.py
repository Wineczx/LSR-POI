import pandas as pd

# 读取训练集
train_df = pd.read_csv('/data/CaiZhuaoXiao/dataset/GB/GB_train.csv')

# 找出POI访问次数并排序
poi_visit_counts = train_df['POI_id'].value_counts()

# 找出访问次数最少的20%的POI
num_cold_pois = int(len(poi_visit_counts) * 0.2)
cold_poi_ids = poi_visit_counts.sort_values().head(num_cold_pois).index

# 读取测试集
test_df = pd.read_csv('/data/CaiZhuaoXiao/dataset/GB/GB_test.csv')

# 将时间列转换为时间格式以便排序
test_df['UTC_time'] = pd.to_datetime(test_df['UTC_time'])

# 按轨迹分组，找到每条轨迹最后访问的POI
last_poi_per_trajectory = test_df.sort_values(by='UTC_time').groupby('trajectory_id').last()

# 找出最后访问的POI在冷POI列表中的轨迹
target_trajectories = last_poi_per_trajectory[last_poi_per_trajectory['POI_id'].isin(cold_poi_ids)].index

# 保留这些轨迹的所有记录
cold_poi_trajectories = test_df[test_df['trajectory_id'].isin(target_trajectories)]

# 保存结果
cold_poi_trajectories.to_csv('/data/CaiZhuaoXiao/dataset/GB/coldpoi.csv', index=False)
