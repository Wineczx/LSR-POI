import pandas as pd

# 读取训练集
train_df = pd.read_csv('/data/CaiZhuaoXiao/dataset/TKY/TKY_train.csv')

# 统计每个用户的访问次数
user_visit_counts = train_df['user_id'].value_counts()

# 计算访问最少的20%的用户数量
num_cold_users = int(len(user_visit_counts) * 0.2)

# 找到访问最少的20%的用户
cold_user_ids = user_visit_counts.sort_values().head(num_cold_users).index

# 读取测试集
test_df = pd.read_csv('/data/CaiZhuaoXiao/dataset/TKY/TKY_test.csv')

# 从测试集中筛选出冷启动用户的数据
cold_user_df = test_df[test_df['user_id'].isin(cold_user_ids)]

# 保存为 colduser.csv
cold_user_df.to_csv('/data/CaiZhuaoXiao/dataset/TKY/colduser.csv', index=False)
