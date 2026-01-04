import logging
import logging
import os
import pathlib
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from define_dataset import TrajectoryDatasetTrain,TrajectoryDatasetVal
from dataloader import load_graph_adj_mtx, load_graph_node_features,load_graph_node_features_geo
from model import GCN, NodeAttnMap, UserEmbeddings, Time2Vec, CategoryEmbeddings, FuseEmbeddings, TransformerModel,Adapter,GeoHashEmbeddings
from param_parser import parameter_parser
from utils import increment_path, calculate_laplacian_matrix, zipdir, top_k_acc_last_timestep, \
    mAP_metric_last_timestep, MRR_metric_last_timestep, maksed_mse_loss,init_torch_seeds,calculate_entropy,ndcg_last_timestep,Contrastive_Loss2
from diversity_metric import DILAD,DCC,FDCC

def train(args):
    args.save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, sep='-')
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    init_torch_seeds(args.seed)
    # Setup logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(args.save_dir, f"log_training.txt"),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True

    # Save run settings
    logging.info(args)
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # Save python code
    zipf = zipfile.ZipFile(os.path.join(args.save_dir, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipdir(pathlib.Path().absolute(), zipf, include_format=['.py'])
    zipf.close()

    # %% ====================== Load data ======================
    # Read check-in train data
    train_df = pd.read_csv(args.data_train)
    val_df = pd.read_csv(args.data_val) 
    # Build POI graph (built from train_df)
    print('Loading POI graph...')
    raw_A = load_graph_adj_mtx(args.data_adj_mtx)
    raw_X ,geohash_list = load_graph_node_features_geo(args.data_node_feats,
                                     args.feature1,
                                     args.feature2,
                                     args.feature3,
                                     args.feature4)

    logging.info(
        f"raw_X.shape: {raw_X.shape}; "
        f"Four features: {args.feature1}, {args.feature2}, {args.feature3}, {args.feature4}.")
    logging.info(f"raw_A.shape: {raw_A.shape}; Edge from row_index to col_index with weight (frequency).")
    num_sim = args.num_sim
    num_pois = raw_X.shape[0]
    num_geos = len(set(geohash_list))
    # One-hot encoding poi categories
    logging.info('One-hot encoding poi categories id')
    one_hot_encoder = OneHotEncoder()
    cat_list = list(raw_X[:, 1])
    one_hot_encoder.fit(list(map(lambda x: [x], cat_list)))
    one_hot_rlt = one_hot_encoder.transform(list(map(lambda x: [x], cat_list))).toarray()
    num_cats = one_hot_rlt.shape[-1]
    X = np.zeros((num_pois, raw_X.shape[-1] - 1 + num_cats), dtype=np.float32)
    X[:, 0] = raw_X[:, 0]
    X[:, 1:num_cats + 1] = one_hot_rlt
    X[:, num_cats + 1:] = raw_X[:, 2:]
    logging.info(f"After one hot encoding poi cat, X.shape: {X.shape}")
    logging.info(f'POI categories: {list(one_hot_encoder.categories_[0])}')
    # Save ont-hot encoder
    with open(os.path.join(args.save_dir, 'one-hot-encoder.pkl'), "wb") as f:
        pickle.dump(one_hot_encoder, f)

    # Normalization
    print('Laplician matrix...')
    A = calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat')

    # POI id to index
    nodes_df = pd.read_csv(args.data_node_feats)
    poi_ids = sorted(list(set(nodes_df['node_name/poi_id'].tolist())))
    poi_id2idx_dict = dict(zip(poi_ids, range(len(poi_ids))))

    # Cat id to index
    cat_ids = sorted(list(set(nodes_df[args.feature2].tolist())))
    cat_id2idx_dict = dict(zip(cat_ids, range(len(cat_ids))))

    # Poi idx to cat idx
    poi_idx2cat_idx_dict = {}
    for i, row in nodes_df.iterrows():
        poi_idx2cat_idx_dict[poi_id2idx_dict[row['node_name/poi_id']]] = \
            cat_id2idx_dict[row[args.feature2]]

        # 初始化 item_categories 列表，假设最大 POI 索引为 max_poi_idx
    max_poi_idx = max(poi_idx2cat_idx_dict.keys()) + 1
    item_categories = [None] * max_poi_idx

    # 根据 poi_idx2cat_idx_dict 填充 item_categories
    for poi_idx, cat_idx in poi_idx2cat_idx_dict.items():
        item_categories[poi_idx] = cat_idx
    
    #地理
    geo_ids = sorted(list(set(geohash_list)))
    poi_idx2geohash_id = {idx: geohash_list[idx] for idx in range(len(geohash_list))}
    # User id to index

    user_ids = sorted(list(set(train_df['user_id'].astype(str).tolist())))
    # 为用户创建索引映射，确保每次顺序一致
    user_id2idx_dict = dict(zip(user_ids, range(len(user_ids))))
    # Print user-trajectories count
    traj_list = list(set(train_df['trajectory_id'].tolist()))
    # 定义文件路径
    file_path = args.data_feature
    # 读取 .pkl 文件
    with open(file_path, "rb") as f:
        poi_features = pickle.load(f)
    feature_dim = 1536  # 假设特征维度为 1536
    features = torch.zeros((num_pois, feature_dim), dtype=torch.float32)  # 初始化特征张量

    # 填充特征张量
    for poi_id, idx in poi_id2idx_dict.items():
        if poi_id in poi_features:
            features[idx] = torch.tensor(poi_features[poi_id], dtype=torch.float32).to(device=args.device)
    # %% ====================== Define dataloader ======================
    print('Prepare dataloader...')
    train_dataset = TrajectoryDatasetTrain(train_df,poi_id2idx_dict,user_id2idx_dict,args)
    val_dataset = TrajectoryDatasetVal(val_df,poi_id2idx_dict,user_id2idx_dict,args)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch,
                              shuffle=True, drop_last=False,
                              pin_memory=True, num_workers=args.workers,
                              collate_fn=lambda x: x,
                                generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch,
                            shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=args.workers,
                            collate_fn=lambda x: x,
                              generator=torch.Generator().manual_seed(args.seed))

    # %% ====================== Build Models ======================
    # Model1: POI embedding model
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
        A = torch.from_numpy(A)
    X = X.to(device=args.device, dtype=torch.float)
    A = A.to(device=args.device, dtype=torch.float)
    features = features.to(device=args.device, dtype=torch.float)
    args.gcn_nfeat = X.shape[1]
    # poi_embed_model = GCN(ninput=args.gcn_nfeat,
    #                       nhid=args.gcn_nhid,
    #                       noutput=args.poi_embed_dim,
    #                       dropout=args.gcn_dropout)
    llm_embed_model = GCN(ninput=1536,
                          nhid=[768,256],
                          noutput=args.poi_embed_dim,
                          dropout=args.gcn_dropout)
    # Node Attn Model
    node_attn_model = NodeAttnMap(in_features=features.shape[1], nhid=args.node_attn_nhid, use_mask=False)

    # %% Model2: User embedding model, nn.embedding
    num_users = len(user_id2idx_dict)
    user_embed_model = UserEmbeddings(num_users, args.user_embed_dim)

    # %% Model3: Time Model
    # time_embed_model = Time2Vec('sin', out_dim=args.time_embed_dim)

    # %% Model4: Category embedding model
    cat_embed_model = CategoryEmbeddings(num_cats, args.cat_embed_dim)
    geo_embed_model=GeoHashEmbeddings(num_geos, 32)
    # %% Model5: Embedding fusion models
    embed_fuse_model1 = FuseEmbeddings(args.user_embed_dim, args.poi_embed_dim)
    embed_fuse_model2 = FuseEmbeddings(32, args.cat_embed_dim)
    # adapter = Adapter(1536, 128)
    # %% Model6: Sequence model
    args.seq_input_embed = args.poi_embed_dim + args.user_embed_dim + args.cat_embed_dim + 32
    seq_model = TransformerModel(num_pois,
                                 num_cats,
                                 num_geos,
                                 args.seq_input_embed,
                                 args.transformer_nhead,
                                 args.transformer_nhid,
                                 args.transformer_nlayers,
                                 dropout=args.transformer_dropout)

    # Define overall loss and optimizer
    optimizer = optim.Adam(params=
    # list(poi_embed_model.parameters()) +
                                  list(node_attn_model.parameters()) +
                                  list(user_embed_model.parameters()) +
                                #   list(time_embed_model.parameters()) +
                                  list(cat_embed_model.parameters()) +
                                  list(geo_embed_model.parameters()) +
                                  list(embed_fuse_model1.parameters()) +
                                  list(embed_fuse_model2.parameters()) +
                                  list(seq_model.parameters())+
                                  list(llm_embed_model.parameters()),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_cat = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_geo = nn.CrossEntropyLoss(ignore_index=-1)
    align_loss = Contrastive_Loss2()
    criterion_time = maksed_mse_loss
    align_kd = nn.MSELoss()
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 'min', verbose=True, factor=args.lr_scheduler_factor)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.9)
    # %% Tool functions for training
    def input_traj_to_embeddings(sample,llm_embeddings):
        # Parse sample
        traj_id = sample[0]
        input_seq = [each[0] for each in sample[1]]
        input_seq_time = [each[1] for each in sample[1]]
        input_seq_cat = [poi_idx2cat_idx_dict[each] for each in input_seq]
        input_seq_geohash = [poi_idx2geohash_id[each] for each in input_seq]
        # User to embedding
        user_id = traj_id.split('_')[0]
        user_idx = user_id2idx_dict[user_id]
        input = torch.LongTensor([user_idx]).to(device=args.device)
        user_embedding = user_embed_model(input)
        user_embedding = torch.squeeze(user_embedding)

        # POI to embedding and fuse embeddings
        input_seq_embed = []
        # Convert input sequence to LLM embeddings
        for idx in range(len(input_seq)):
            # poi_embedding = poi_embeddings[input_seq[idx]]
            # poi_embedding = torch.squeeze(poi_embedding).to(device=args.device)
            llm_embedding = llm_embeddings[input_seq[idx]]
            llm_embedding = torch.squeeze(llm_embedding).to(device=args.device)
            # Time to vector
            # time_embedding = time_embed_model(
            #     torch.tensor([input_seq_time[idx]], dtype=torch.float).to(device=args.device))
            # time_embedding = torch.squeeze(time_embedding).to(device=args.device)

            # Categroy to embedding
            cat_idx = torch.LongTensor([input_seq_cat[idx]]).to(device=args.device)
            cat_embedding = cat_embed_model(cat_idx)
            cat_embedding = torch.squeeze(cat_embedding)
            geohash_idx = torch.LongTensor([input_seq_geohash[idx]]).to(device=args.device)
            geohash_embedding = geo_embed_model(geohash_idx)
            geohash_embedding = torch.squeeze(geohash_embedding)
            # Fuse user+poi embeds
            fused_embedding1 = embed_fuse_model1(user_embedding, llm_embedding)
            # fused_embedding2 = torch.cat((llm_embedding, cat_embedding), dim=-1)
            fused_embedding2 = embed_fuse_model2(geohash_embedding, cat_embedding)
            # Concat time, cat after user+poi
            concat_embedding = torch.cat((fused_embedding1, fused_embedding2), dim=-1)
            # Save final embed
            input_seq_embed.append(concat_embedding)
        return input_seq_embed

    def adjust_pred_prob_by_graph(y_pred_poi):
        y_pred_poi_adjusted = torch.zeros_like(y_pred_poi)
        attn_map = node_attn_model(features, A)

        for i in range(len(batch_seq_lens)):
            traj_i_input = batch_input_seqs[i]  # list of input check-in pois
            for j in range(len(traj_i_input)):
                y_pred_poi_adjusted[i, j, :] = attn_map[traj_i_input[j], :] + y_pred_poi[i, j, :]

        return y_pred_poi_adjusted

    # %% ====================== Train ======================
    # poi_embed_model = poi_embed_model.to(device=args.device)
    node_attn_model = node_attn_model.to(device=args.device)
    user_embed_model = user_embed_model.to(device=args.device)
    # time_embed_model = time_embed_model.to(device=args.device)
    cat_embed_model = cat_embed_model.to(device=args.device)
    embed_fuse_model1 = embed_fuse_model1.to(device=args.device)
    embed_fuse_model2 = embed_fuse_model2.to(device=args.device)
    seq_model = seq_model.to(device=args.device)
    llm_embed_model = llm_embed_model.to(device=args.device)
    geo_embed_model = geo_embed_model.to(device=args.device)
    # %% Loop epoch
    # %%For plotting
    train_epochs_top1_acc_list = []
    train_epochs_top5_acc_list = []
    train_epochs_top10_acc_list = []
    train_epochs_top20_acc_list = []
    train_epochs_ng1_acc_list = []
    train_epochs_ng5_acc_list = []
    train_epochs_ng10_acc_list = []
    train_epochs_ng20_acc_list = []
    train_epochs_mAP20_list = []
    train_epochs_mrr_list = []
    train_epochs_loss_list = []
    train_epochs_poi_loss_list = []
    train_epochs_geo_loss_list = []
    train_epochs_cat_loss_list = []
    train_epochs_dilad_list = []
    train_epochs_dcc_list = []
    train_epochs_fdcc_list = []
    val_epochs_top1_acc_list = []
    val_epochs_top5_acc_list = []
    val_epochs_top10_acc_list = []
    val_epochs_top20_acc_list = []
    val_epochs_ng1_acc_list = []
    val_epochs_ng5_acc_list = []
    val_epochs_ng10_acc_list = []
    val_epochs_ng20_acc_list = []
    val_epochs_mAP20_list = []
    val_epochs_mrr_list = []
    val_epochs_loss_list = []
    val_epochs_poi_loss_list = []
    val_epochs_geo_loss_list = []
    val_epochs_cat_loss_list = []
    val_epochs_dilad_list = []
    val_epochs_dcc_list = []
    val_epochs_fdcc_list = []
    # %%For saving ckpt
    max_val_score = -np.inf

    for epoch in range(args.epochs):
        logging.info(f"{'*' * 50}Epoch:{epoch:03d}{'*' * 50}\n")
        # poi_embed_model.train()
        node_attn_model.train()
        user_embed_model.train()
        # time_embed_model.train()
        cat_embed_model.train()
        embed_fuse_model1.train()
        embed_fuse_model2.train()
        seq_model.train()
        llm_embed_model.train()
        geo_embed_model.train()
        train_batches_top1_acc_list = []
        train_batches_top5_acc_list = []
        train_batches_top10_acc_list = []
        train_batches_top20_acc_list = []
        train_batches_ng1_acc_list = []
        train_batches_ng5_acc_list = []
        train_batches_ng10_acc_list = []
        train_batches_ng20_acc_list = []
        train_batches_mAP20_list = []
        train_batches_mrr_list = []
        train_batches_loss_list = []
        train_batches_poi_loss_list = []
        train_batches_geo_loss_list = []
        train_batches_cat_loss_list = []
        batch_dilad = []
        batch_dcc = []
        batch_fdcc = []
        src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device)
        src_mask2 = seq_model.generate_square_subsequent_mask(args.batch*num_sim).to(args.device)
        # Loop batch
        for b_idx, batch in enumerate(train_loader):
            if len(batch) != args.batch:
                src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)
                src_mask2 = seq_model.generate_square_subsequent_mask(len(batch)*num_sim).to(args.device)

            # For padding
            batch_input_seqs = []
            batch_traj = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_seq_labels_poi = []
            batch_seq_labels_time = []
            batch_seq_labels_cat = []
            batch_seq_labels_geo = []
            batch_similar_seq_embeds = []  # 新增：用于存储相似轨迹的特征
            # poi_embeddings = poi_embed_model(X, A)
            llm_embeddings = llm_embed_model(features,A)
            # Convert input seq to embeddings
            for sample in batch:
                # sample[0]: traj_id, sample[1]: input_seq, sample[2]: label_seq
                traj_id = sample[0]
                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                input_seq_time = [each[1] for each in sample[1]]
                label_seq_time = [each[1] for each in sample[2]]
                label_seq_cats = [poi_idx2cat_idx_dict[each] for each in label_seq]
                label_seq_geos = [poi_idx2geohash_id[each] for each in label_seq]
                input_seq_embed = torch.stack(input_traj_to_embeddings(sample,llm_embeddings))
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
                batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))
                batch_seq_labels_geo.append(torch.LongTensor(label_seq_geos))
                    # 相似轨迹的嵌入
                similar_trajs = sample[3]
                similar_traj_embeds = []
                for similar_traj in similar_trajs:
                    similar_traj_id = similar_traj["traj_id"]
                    similar_input_seq = similar_traj["full_traj"]  # 完整轨迹数据
                    # 使用 input_traj_to_embeddings 函数计算嵌入
                    similar_input_seq_embed = torch.stack(input_traj_to_embeddings((similar_traj_id, similar_input_seq), llm_embeddings))
                    similar_traj_embeds.append(similar_input_seq_embed)
                # 对于每个样本，将其相似轨迹的嵌入作为附加信息存储
                batch_similar_seq_embeds.append(similar_traj_embeds)

            # 假设 batch_similar_seq_embeds 是形状为 [batch, num_sim, seq_len, dim]
            # 1. 计算批次中的最大序列长度
            max_seq_len = max([max([len(seq) for seq in similar_traj_embeds]) for similar_traj_embeds in batch_similar_seq_embeds])

            # 2. 对 batch_similar_seq_embeds 进行填充
            padded_similar_seq_embeds = []
            for similar_traj_embeds in batch_similar_seq_embeds:
                padded_similar_traj_embeds = []
                for traj_embed in similar_traj_embeds:
                    pad_len = max_seq_len - len(traj_embed)
                    # 用 -1 填充轨迹，确保填充到最大序列长度
                    padded_traj_embed = torch.cat([traj_embed, torch.full((pad_len, traj_embed.shape[1]), -1, device=args.device)], dim=0)
                    padded_similar_traj_embeds.append(padded_traj_embed)
                padded_similar_seq_embeds.append(padded_similar_traj_embeds)

            # 3. 合并所有相似轨迹的嵌入，转换为形状 [batch*num_sim, max_seq_len, dim]
            all_similar_embeds = torch.stack([torch.stack(similar_traj_embeds) for similar_traj_embeds in padded_similar_seq_embeds], dim=0)  # shape: [batch, num_sim, max_seq_len, dim]
            all_similar_embeds = all_similar_embeds.view(-1, max_seq_len, all_similar_embeds.shape[-1])  # shape: [batch*num_sim, max_seq_len, dim]

            # 4. 获取每个序列的有效长度
            actual_seq_len = torch.sum(all_similar_embeds != -1, dim=1)  # shape: [batch*num_sim]
            actual_seq_len = actual_seq_len[:, 0]
            # 5. 提取每个轨迹的最后一个有效嵌入
            x_sim = all_similar_embeds.to(device=args.device, dtype=torch.float)
            y_sim_poi,  y_sim_cat ,y_sim_geo,xu_sim= seq_model(x_sim, src_mask2)
            xu_sim = xu_sim.detach().to(device=args.device)
            last_embeddings = []
            for i in range(xu_sim.shape[0]):
                traj_embed = xu_sim[i]  # [max_seq_len, dim]
                traj_len = actual_seq_len[i]  # 当前轨迹的有效长度
                # 使用有效长度来获取最后一个有效的嵌入
                last_embed = traj_embed[traj_len - 1]  # 取有效长度位置的嵌入
                last_embeddings.append(last_embed)

            # 将最后的嵌入转换为 [batch*num_sim, dim] 的形状
            last_embeddings = torch.stack(last_embeddings)  # shape: [batch*num_sim, dim]

            # 6. 将相似轨迹的最后一个嵌入重塑为 [batch, num_sim, dim]
            last_embeddings = last_embeddings.view(len(batch_similar_seq_embeds), num_sim, -1)  # shape: [batch, num_sim, dim]

            # 7. 计算相似轨迹的均值 [batch, dim]
            mean_embeddings = last_embeddings.mean(dim=1)  # shape: [batch, dim]
            # Pad seqs for batch training
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
            # label_padded_time = pad_sequence(batch_seq_labels_time, batch_first=True, padding_value=-1)
            label_padded_cat = pad_sequence(batch_seq_labels_cat, batch_first=True, padding_value=-1)
            label_padded_geo = pad_sequence(batch_seq_labels_geo, batch_first=True, padding_value=-1)
            # 获取每个序列的实际长度 (注意: 需要提前保存每个序列的实际长度)
            seq_lengths = torch.tensor([len(seq) for seq in batch_seq_embeds])  # 每个序列的真实长度
            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            # y_time = label_padded_time.to(device=args.device, dtype=torch.float)
            y_cat = label_padded_cat.to(device=args.device, dtype=torch.long)
            y_geo = label_padded_geo.to(device=args.device, dtype=torch.long)
            y_pred_poi,  y_pred_cat ,y_pred_geo,xu= seq_model(x, src_mask)
            # 提取每个序列的最后一个有效特征
            batch_indices = torch.arange(xu.size(0)).to(device=args.device)  # 每个 batch 的索引
            last_indices = (seq_lengths - 1).to(device=args.device)  # 每个序列的最后有效位置
            last_features = xu[batch_indices, last_indices]  # 提取最后一个有效特征
            # Graph Attention adjusted prob
            y_pred_poi_adjusted = adjust_pred_prob_by_graph(y_pred_poi)
            torch.use_deterministic_algorithms(False)
            loss_cl = align_loss(last_features,mean_embeddings)
            loss_poi = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), y_poi)
            # loss_time = criterion_time(torch.squeeze(y_pred_time), y_time)
            loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat)
            loss_geo = criterion_geo(y_pred_geo.transpose(1, 2), y_geo)
            torch.use_deterministic_algorithms(True)
            # Final loss
            loss = loss_poi + loss_cat+loss_geo+loss_cl*args.sim_loss_weight
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            #%% Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            ng1_acc = 0
            ng5_acc = 0
            ng10_acc = 0
            ng20_acc = 0
            mAP20 = 0
            mrr = 0
            all_top_k_rec = []  # 用于存储所有用户的推荐列表
            all_y_true = []     # 用于存储所有用户的真实值  
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi_adjusted.detach().cpu().numpy()
            # batch_pred_times = y_pred_time.detach().cpu().numpy()
            batch_pred_cats = y_pred_cat.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
                label_pois = label_pois[:seq_len]  # shape: (seq_len, )
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                ng1_acc += ndcg_last_timestep(label_pois, pred_pois, k=1)
                ng5_acc += ndcg_last_timestep(label_pois, pred_pois, k=5)
                ng10_acc += ndcg_last_timestep(label_pois, pred_pois, k=10)
                ng20_acc += ndcg_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
                # 计算单用户的 Top-K 推荐列表
                y_true = label_pois[-1]  # 取最后一个 POI 的真实值
                y_pred = pred_pois[-1]  # 取最后一个时间步的预测值
                top_k_rec = y_pred.argsort()[-10:][::-1]  # 获取 Top-K 索引

                # 将当前用户的推荐和真实值添加到全局列表中
                all_top_k_rec.append(top_k_rec)
                all_y_true.append([y_true])  # 包装为列表以保持二维格式
            # 计算DILAD、DCC、FDCC
            all_top_k_rec = np.array(all_top_k_rec)  # shape: (batch_size, k)
            all_y_true = np.array(all_y_true)       # shape: (batch_size, 1)
            # 计算 DILAD、DCC、FDCC
            item_feat = llm_embed_model(features, A).detach().cpu().numpy()
            dilad_value = DILAD(all_top_k_rec, all_y_true, item_feat, alpha=0.1)
            dcc_value = DCC(all_top_k_rec, all_y_true, item_categories, num_cats, alpha=0.1)
            fdcc_value = FDCC(all_top_k_rec, all_y_true, item_categories, num_cats, alpha=0.1)

            batch_dilad.append(dilad_value/ len(batch_label_pois))
            batch_dcc.append(dcc_value/ len(batch_label_pois))
            batch_fdcc.append(fdcc_value/ len(batch_label_pois))
            train_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            train_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            train_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            train_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            train_batches_ng1_acc_list.append(ng1_acc / len(batch_label_pois))
            train_batches_ng5_acc_list.append(ng5_acc / len(batch_label_pois))
            train_batches_ng10_acc_list.append(ng10_acc / len(batch_label_pois))
            train_batches_ng20_acc_list.append(ng20_acc / len(batch_label_pois))
            train_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            train_batches_mrr_list.append(mrr / len(batch_label_pois))
            train_batches_loss_list.append(loss.detach().cpu().numpy())
            train_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            train_batches_geo_loss_list.append(loss_geo.detach().cpu().numpy())
            train_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())

            # Report training progress
            if (b_idx % (args.batch * 5)) == 0:
                sample_idx = 0
                batch_pred_pois_wo_attn = y_pred_poi.detach().cpu().numpy()
                logging.info(f'Epoch:{epoch}, batch:{b_idx}, '
                             f'train_batch_loss:{loss.item():.2f}, '
                             f'train_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                             f'train_move_loss:{np.mean(train_batches_loss_list):.2f}\n'
                             f'train_move_poi_loss:{np.mean(train_batches_poi_loss_list):.2f}\n'
                             f'dilad_value:{np.mean(batch_dilad):.4f}\n'
                             f'dcc_value:{np.mean(batch_dcc):.4f}\n'
                             f'fdcc_value:{np.mean(batch_fdcc):.4f}\n'
                             f'train_move_geo_loss:{np.mean(train_batches_geo_loss_list):.2f}\n'
                             f'train_move_top1_acc:{np.mean(train_batches_top1_acc_list):.4f}\n'
                             f'train_move_top5_acc:{np.mean(train_batches_top5_acc_list):.4f}\n'
                             f'train_move_top10_acc:{np.mean(train_batches_top10_acc_list):.4f}\n'
                             f'train_move_top20_acc:{np.mean(train_batches_top20_acc_list):.4f}\n'
                            f'train_move_ng1_acc:{np.mean(train_batches_ng1_acc_list):.4f}\n'
                             f'train_move_ng5_acc:{np.mean(train_batches_ng5_acc_list):.4f}\n'
                             f'train_move_ng10_acc:{np.mean(train_batches_ng10_acc_list):.4f}\n'
                             f'train_move_ng20_acc:{np.mean(train_batches_ng20_acc_list):.4f}\n'
                             f'train_move_mAP20:{np.mean(train_batches_mAP20_list):.4f}\n'
                             f'train_move_MRR:{np.mean(train_batches_mrr_list):.4f}\n'
                             f'traj_id:{batch[sample_idx][0]}\n'
                             f'input_seq: {batch[sample_idx][1]}\n'
                             f'label_seq:{batch[sample_idx][2]}\n'
                             f'pred_seq_poi_wo_attn:{list(np.argmax(batch_pred_pois_wo_attn, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'label_seq_cat:{[poi_idx2cat_idx_dict[each[0]] for each in batch[sample_idx][2]]}\n'
                             f'pred_seq_cat:{list(np.argmax(batch_pred_cats, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n' +
                             '=' * 100)

        # %%train end --------------------------------------------------------------------------------------------------------
        # poi_embed_model.eval()
        node_attn_model.eval()
        user_embed_model.eval()
        # time_embed_model.eval()
        cat_embed_model.eval()
        embed_fuse_model1.eval()
        embed_fuse_model2.eval()
        seq_model.eval()
        llm_embed_model.eval()
        geo_embed_model.eval()
        val_batches_top1_acc_list = []
        val_batches_top5_acc_list = []
        val_batches_top10_acc_list = []
        val_batches_top20_acc_list = []
        val_batches_ng1_acc_list = []
        val_batches_ng5_acc_list = []
        val_batches_ng10_acc_list = []
        val_batches_ng20_acc_list = []
        val_batches_mAP20_list = []
        val_batches_mrr_list = []
        val_batches_loss_list = []
        val_batches_poi_loss_list = []
        val_batches_geo_loss_list = []
        val_batches_cat_loss_list = []
        val_batch_dilad = []
        val_batch_dcc = []
        val_batch_fdcc = []
        src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device)
        for vb_idx, batch in enumerate(val_loader):
            if len(batch) != args.batch:
                src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)

            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_seq_labels_poi = []
            batch_seq_labels_time = []
            batch_seq_labels_cat = []
            batch_seq_labels_geo = []
            # poi_embeddings = poi_embed_model(X, A)
            llm_embeddings = llm_embed_model(features,A)    
            # Convert input seq to embeddings
            for sample in batch:
                traj_id = sample[0]
                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                input_seq_time = [each[1] for each in sample[1]]
                label_seq_time = [each[1] for each in sample[2]]
                label_seq_cats = [poi_idx2cat_idx_dict[each] for each in label_seq]
                label_seq_geos = [poi_idx2geohash_id[each] for each in label_seq]
                input_seq_embed = torch.stack(input_traj_to_embeddings(sample,llm_embeddings))
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
                batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))
                batch_seq_labels_geo.append(torch.LongTensor(label_seq_geos))
            # Pad seqs for batch training
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
            # label_padded_time = pad_sequence(batch_seq_labels_time, batch_first=True, padding_value=-1)
            label_padded_cat = pad_sequence(batch_seq_labels_cat, batch_first=True, padding_value=-1)
            label_padded_geo = pad_sequence(batch_seq_labels_geo, batch_first=True, padding_value=-1)
            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            # y_time = label_padded_time.to(device=args.device, dtype=torch.float)
            y_cat = label_padded_cat.to(device=args.device, dtype=torch.long)
            y_geo = label_padded_geo.to(device=args.device, dtype=torch.long)
            y_pred_poi,  y_pred_cat ,y_pred_geo,xu = seq_model(x, src_mask)


            # Graph Attention adjusted prob
            y_pred_poi_adjusted = adjust_pred_prob_by_graph(y_pred_poi)

            # Calculate loss
            torch.use_deterministic_algorithms(False)
            loss_poi = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), y_poi)
            # loss_time = criterion_time(torch.squeeze(y_pred_time), y_time)
            loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat)
            loss_geo = criterion_geo(y_pred_geo.transpose(1, 2), y_geo)
            torch.use_deterministic_algorithms(True)
            loss = loss_poi+ loss_cat +loss_geo

            # %%Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            ng1_acc = 0
            ng5_acc = 0
            ng10_acc = 0
            ng20_acc = 0
            mAP20 = 0
            mrr = 0
            all_top_k_rec = []  # 用于存储所有用户的推荐列表
            all_y_true = []     # 用于存储所有用户的真实值  
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi_adjusted.detach().cpu().numpy()
            # batch_pred_times = y_pred_time.detach().cpu().numpy()
            batch_pred_cats = y_pred_cat.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
                label_pois = label_pois[:seq_len]  # shape: (seq_len, )
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                ng1_acc += ndcg_last_timestep(label_pois, pred_pois, k=1)
                ng5_acc += ndcg_last_timestep(label_pois, pred_pois, k=5)
                ng10_acc += ndcg_last_timestep(label_pois, pred_pois, k=10)
                ng20_acc += ndcg_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
                # 计算单用户的 Top-K 推荐列表
                y_true = label_pois[-1]  # 取最后一个 POI 的真实值
                y_pred = pred_pois[-1]  # 取最后一个时间步的预测值
                top_k_rec = y_pred.argsort()[-10:][::-1]  # 获取 Top-K 索引

                # 将当前用户的推荐和真实值添加到全局列表中
                all_top_k_rec.append(top_k_rec)
                all_y_true.append([y_true])  # 包装为列表以保持二维格式
            # 计算DILAD、DCC、FDCC
            all_top_k_rec = np.array(all_top_k_rec)  # shape: (batch_size, k)
            all_y_true = np.array(all_y_true)       # shape: (batch_size, 1)
            # 计算 DILAD、DCC、FDCC
            item_feat = llm_embed_model(features, A).detach().cpu().numpy()
            dilad_value = DILAD(all_top_k_rec, all_y_true, item_feat, alpha=0.1)
            dcc_value = DCC(all_top_k_rec, all_y_true, item_categories, num_cats, alpha=0.1)
            fdcc_value = FDCC(all_top_k_rec, all_y_true, item_categories, num_cats, alpha=0.1)

            val_batch_dilad.append(dilad_value/ len(batch_label_pois))
            val_batch_dcc.append(dcc_value/ len(batch_label_pois))
            val_batch_fdcc.append(fdcc_value/ len(batch_label_pois))
            val_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            val_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            val_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            val_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            val_batches_ng1_acc_list.append(ng1_acc / len(batch_label_pois))
            val_batches_ng5_acc_list.append(ng5_acc / len(batch_label_pois))
            val_batches_ng10_acc_list.append(ng10_acc / len(batch_label_pois))
            val_batches_ng20_acc_list.append(ng20_acc / len(batch_label_pois))
            val_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            val_batches_mrr_list.append(mrr / len(batch_label_pois))
            val_batches_loss_list.append(loss.detach().cpu().numpy())
            val_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            val_batches_geo_loss_list.append(loss_geo.detach().cpu().numpy())
            val_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())

            # Report validation progress
            if (vb_idx % (args.batch * 2)) == 0:
                sample_idx = 0
                batch_pred_pois_wo_attn = y_pred_poi.detach().cpu().numpy()
                logging.info(f'Epoch:{epoch}, batch:{vb_idx}, '
                             f'val_batch_loss:{loss.item():.2f}, '
                             f'val_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                             f'val_move_loss:{np.mean(val_batches_loss_list):.2f} \n'
                             f'val_move_poi_loss:{np.mean(val_batches_poi_loss_list):.2f} \n'
                             f'dilad_value:{np.mean(val_batch_dilad):.4f}\n'
                             f'dcc_value:{np.mean(val_batch_dcc):.4f}\n'
                             f'fdcc_value:{np.mean(val_batch_fdcc):.4f}\n'
                             f'val_move_geo_loss:{np.mean(val_batches_geo_loss_list):.2f} \n'
                             f'val_move_top1_acc:{np.mean(val_batches_top1_acc_list):.4f} \n'
                             f'val_move_top5_acc:{np.mean(val_batches_top5_acc_list):.4f} \n'
                             f'val_move_top10_acc:{np.mean(val_batches_top10_acc_list):.4f} \n'
                             f'val_move_top20_acc:{np.mean(val_batches_top20_acc_list):.4f} \n'
                            f'val_move_ng1_acc:{np.mean(val_batches_ng1_acc_list):.4f} \n'
                             f'val_move_ng5_acc:{np.mean(val_batches_ng5_acc_list):.4f} \n'
                             f'val_move_ng10_acc:{np.mean(val_batches_ng10_acc_list):.4f} \n'
                             f'val_move_ng20_acc:{np.mean(val_batches_ng20_acc_list):.4f} \n'
                             f'val_move_mAP20:{np.mean(val_batches_mAP20_list):.4f} \n'
                             f'val_move_MRR:{np.mean(val_batches_mrr_list):.4f} \n'
                             f'traj_id:{batch[sample_idx][0]}\n'
                             f'input_seq:{batch[sample_idx][1]}\n'
                             f'label_seq:{batch[sample_idx][2]}\n'
                             f'pred_seq_poi_wo_attn:{list(np.argmax(batch_pred_pois_wo_attn, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'label_seq_cat:{[poi_idx2cat_idx_dict[each[0]] for each in batch[sample_idx][2]]}\n'
                             f'pred_seq_cat:{list(np.argmax(batch_pred_cats, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'+
                             '=' * 100)
        # %%valid end --------------------------------------------------------------------------------------------------------

        # Calculate epoch metrics
        epoch_train_dilad = np.mean(batch_dilad)
        epoch_train_dcc = np.mean(batch_dcc)
        epoch_train_fdcc = np.mean(batch_fdcc)
        epoch_train_top1_acc = np.mean(train_batches_top1_acc_list)
        epoch_train_top5_acc = np.mean(train_batches_top5_acc_list)
        epoch_train_top10_acc = np.mean(train_batches_top10_acc_list)
        epoch_train_top20_acc = np.mean(train_batches_top20_acc_list)
        epoch_train_ng1_acc = np.mean(train_batches_ng1_acc_list)
        epoch_train_ng5_acc = np.mean(train_batches_ng5_acc_list)
        epoch_train_ng10_acc = np.mean(train_batches_ng10_acc_list)
        epoch_train_ng20_acc = np.mean(train_batches_ng20_acc_list)
        epoch_train_mAP20 = np.mean(train_batches_mAP20_list)
        epoch_train_mrr = np.mean(train_batches_mrr_list)
        epoch_train_loss = np.mean(train_batches_loss_list)
        epoch_train_poi_loss = np.mean(train_batches_poi_loss_list)
        epoch_train_geo_loss = np.mean(train_batches_geo_loss_list)
        epoch_train_cat_loss = np.mean(train_batches_cat_loss_list)

        epoch_val_dilad = np.mean(val_batch_dilad)
        epoch_val_dcc = np.mean(val_batch_dcc)
        epoch_val_fdcc = np.mean(val_batch_fdcc)
        epoch_val_top1_acc = np.mean(val_batches_top1_acc_list)
        epoch_val_top5_acc = np.mean(val_batches_top5_acc_list)
        epoch_val_top10_acc = np.mean(val_batches_top10_acc_list)
        epoch_val_top20_acc = np.mean(val_batches_top20_acc_list)
        epoch_val_ng1_acc = np.mean(val_batches_ng1_acc_list)
        epoch_val_ng5_acc = np.mean(val_batches_ng5_acc_list)
        epoch_val_ng10_acc = np.mean(val_batches_ng10_acc_list)
        epoch_val_ng20_acc = np.mean(val_batches_ng20_acc_list)
        epoch_val_mAP20 = np.mean(val_batches_mAP20_list)
        epoch_val_mrr = np.mean(val_batches_mrr_list)
        epoch_val_loss = np.mean(val_batches_loss_list)
        epoch_val_poi_loss = np.mean(val_batches_poi_loss_list)
        epoch_val_geo_loss = np.mean(val_batches_geo_loss_list)
        epoch_val_cat_loss = np.mean(val_batches_cat_loss_list)

        # Save metrics to list
        train_epochs_loss_list.append(epoch_train_loss)
        train_epochs_poi_loss_list.append(epoch_train_poi_loss)
        train_epochs_geo_loss_list.append(epoch_train_geo_loss)
        train_epochs_cat_loss_list.append(epoch_train_cat_loss)
        train_epochs_top1_acc_list.append(epoch_train_top1_acc)
        train_epochs_top5_acc_list.append(epoch_train_top5_acc)
        train_epochs_top10_acc_list.append(epoch_train_top10_acc)
        train_epochs_top20_acc_list.append(epoch_train_top20_acc)
        train_epochs_ng1_acc_list.append(epoch_train_ng1_acc)
        train_epochs_ng5_acc_list.append(epoch_train_ng5_acc)
        train_epochs_ng10_acc_list.append(epoch_train_ng10_acc)
        train_epochs_ng20_acc_list.append(epoch_train_ng20_acc)
        train_epochs_mAP20_list.append(epoch_train_mAP20)
        train_epochs_mrr_list.append(epoch_train_mrr)
        train_epochs_dilad_list.append(epoch_train_dilad)
        train_epochs_dcc_list.append(epoch_train_dcc)
        train_epochs_fdcc_list.append(epoch_train_fdcc)

        val_epochs_loss_list.append(epoch_val_loss)
        val_epochs_poi_loss_list.append(epoch_val_poi_loss)
        val_epochs_geo_loss_list.append(epoch_val_geo_loss)
        val_epochs_cat_loss_list.append(epoch_val_cat_loss)
        val_epochs_top1_acc_list.append(epoch_val_top1_acc)
        val_epochs_top5_acc_list.append(epoch_val_top5_acc)
        val_epochs_top10_acc_list.append(epoch_val_top10_acc)
        val_epochs_top20_acc_list.append(epoch_val_top20_acc)
        val_epochs_ng1_acc_list.append(epoch_val_ng1_acc)
        val_epochs_ng5_acc_list.append(epoch_val_ng5_acc)
        val_epochs_ng10_acc_list.append(epoch_val_ng10_acc)
        val_epochs_ng20_acc_list.append(epoch_val_ng20_acc)
        val_epochs_mAP20_list.append(epoch_val_mAP20)
        val_epochs_mrr_list.append(epoch_val_mrr)
        val_epochs_dilad_list.append(epoch_val_dilad)
        val_epochs_dcc_list.append(epoch_val_dcc)
        val_epochs_fdcc_list.append(epoch_val_fdcc)
        # Monitor loss and score
        monitor_loss = epoch_val_loss
        monitor_score = np.mean(epoch_val_top1_acc * 4 + epoch_val_top20_acc)

        # Learning rate schuduler
        # lr_scheduler.step(monitor_loss)
        lr_scheduler.step()
        # Save model state dict
        if args.save_weights:
            state_dict = {
                'epoch': epoch,
                # 'poi_embed_state_dict': poi_embed_model.state_dict(),
                # 'node_attn_state_dict': node_attn_model.state_dict(),
                'user_embed_state_dict': user_embed_model.state_dict(),
                # 'time_embed_state_dict': time_embed_model.state_dict(),
                'cat_embed_state_dict': cat_embed_model.state_dict(),
                'embed_fuse1_state_dict': embed_fuse_model1.state_dict(),
                # 'embed_fuse2_state_dict': embed_fuse_model2.state_dict(),
                'seq_model_state_dict': seq_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'user_id2idx_dict': user_id2idx_dict,
                'poi_id2idx_dict': poi_id2idx_dict,
                'cat_id2idx_dict': cat_id2idx_dict,
                'poi_idx2cat_idx_dict': poi_idx2cat_idx_dict,
                # 'node_attn_map': node_attn_model(X, A),
                'args': args,
                'epoch_train_metrics': {
                    'epoch_train_loss': epoch_train_loss,
                    'epoch_train_poi_loss': epoch_train_poi_loss,
                    # 'epoch_train_time_loss': epoch_train_time_loss,
                    'epoch_train_cat_loss': epoch_train_cat_loss,
                    'epoch_train_top1_acc': epoch_train_top1_acc,
                    'epoch_train_top5_acc': epoch_train_top5_acc,
                    'epoch_train_top10_acc': epoch_train_top10_acc,
                    'epoch_train_top20_acc': epoch_train_top20_acc,
                    'epoch_train_mAP20': epoch_train_mAP20,
                    'epoch_train_mrr': epoch_train_mrr
                },
                'epoch_val_metrics': {
                    'epoch_val_loss': epoch_val_loss,
                    'epoch_val_poi_loss': epoch_val_poi_loss,
                    # 'epoch_val_time_loss': epoch_val_time_loss,
                    'epoch_val_cat_loss': epoch_val_cat_loss,
                    'epoch_val_top1_acc': epoch_val_top1_acc,
                    'epoch_val_top5_acc': epoch_val_top5_acc,
                    'epoch_val_top10_acc': epoch_val_top10_acc,
                    'epoch_val_top20_acc': epoch_val_top20_acc,
                    'epoch_val_mAP20': epoch_val_mAP20,
                    'epoch_val_mrr': epoch_val_mrr
                }
            }
            model_save_dir = os.path.join(args.save_dir, 'checkpoints')
            # Save best val score epoch
            if monitor_score >= max_val_score:
                if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
                torch.save(state_dict, rf"{model_save_dir}/best_epoch.state.pt")
                with open(rf"{model_save_dir}/best_epoch.txt", 'w') as f:
                    print(state_dict['epoch_val_metrics'], file=f)
                max_val_score = monitor_score

        # %%Save train/val metrics for plotting purpose
        with open(os.path.join(args.save_dir, 'metrics-train.txt'), "w") as f:
            print(f'train_epochs_loss_list={[float(f"{each:.4f}") for each in train_epochs_loss_list]}', file=f)
            print(f'train_epochs_poi_loss_list={[float(f"{each:.4f}") for each in train_epochs_poi_loss_list]}', file=f)
            print(f'train_epochs_geo_loss_list={[float(f"{each:.4f}") for each in train_epochs_geo_loss_list]}',
                  file=f)
            print(f'train_epochs_cat_loss_list={[float(f"{each:.4f}") for each in train_epochs_cat_loss_list]}', file=f)
            print(f'train_epochs_dilad_list={[float(f"{each:.4f}") for each in train_epochs_dilad_list]}', file=f)
            print(f'train_epochs_dcc_list={[float(f"{each:.4f}") for each in train_epochs_dcc_list]}', file=f)
            print(f'train_epochs_fdcc_list={[float(f"{each:.4f}") for each in train_epochs_fdcc_list]}', file=f)
            print(f'train_epochs_top1_acc_list={[float(f"{each:.4f}") for each in train_epochs_top1_acc_list]}', file=f)
            print(f'train_epochs_top5_acc_list={[float(f"{each:.4f}") for each in train_epochs_top5_acc_list]}', file=f)
            print(f'train_epochs_top10_acc_list={[float(f"{each:.4f}") for each in train_epochs_top10_acc_list]}',
                  file=f)
            print(f'train_epochs_top20_acc_list={[float(f"{each:.4f}") for each in train_epochs_top20_acc_list]}',
                  file=f)
            print(f'train_epochs_ng1_acc_list={[float(f"{each:.4f}") for each in train_epochs_ng1_acc_list]}', file=f)
            print(f'train_epochs_ng5_acc_list={[float(f"{each:.4f}") for each in train_epochs_ng5_acc_list]}', file=f)
            print(f'train_epochs_ng10_acc_list={[float(f"{each:.4f}") for each in train_epochs_ng10_acc_list]}',
                  file=f)
            print(f'train_epochs_ng20_acc_list={[float(f"{each:.4f}") for each in train_epochs_ng20_acc_list]}',
                  file=f)
            print(f'train_epochs_mAP20_list={[float(f"{each:.4f}") for each in train_epochs_mAP20_list]}', file=f)
            print(f'train_epochs_mrr_list={[float(f"{each:.4f}") for each in train_epochs_mrr_list]}', file=f)
        with open(os.path.join(args.save_dir, 'metrics-val.txt'), "w") as f:
            print(f'val_epochs_loss_list={[float(f"{each:.4f}") for each in val_epochs_loss_list]}', file=f)
            print(f'val_epochs_poi_loss_list={[float(f"{each:.4f}") for each in val_epochs_poi_loss_list]}', file=f)
            print(f'val_epochs_geo_loss_list={[float(f"{each:.4f}") for each in val_epochs_geo_loss_list]}', file=f)
            print(f'val_epochs_cat_loss_list={[float(f"{each:.4f}") for each in val_epochs_cat_loss_list]}', file=f)
            print(f'val_epochs_dilad_list={[float(f"{each:.4f}") for each in val_epochs_dilad_list]}', file=f)
            print(f'val_epochs_dcc_list={[float(f"{each:.4f}") for each in val_epochs_dcc_list]}', file=f)
            print(f'val_epochs_fdcc_list={[float(f"{each:.4f}") for each in val_epochs_fdcc_list]}', file=f)
            print(f'val_epochs_top1_acc_list={[float(f"{each:.4f}") for each in val_epochs_top1_acc_list]}', file=f)
            print(f'val_epochs_top5_acc_list={[float(f"{each:.4f}") for each in val_epochs_top5_acc_list]}', file=f)
            print(f'val_epochs_top10_acc_list={[float(f"{each:.4f}") for each in val_epochs_top10_acc_list]}', file=f)
            print(f'val_epochs_top20_acc_list={[float(f"{each:.4f}") for each in val_epochs_top20_acc_list]}', file=f)
            print(f'val_epochs_ng1_acc_list={[float(f"{each:.4f}") for each in val_epochs_ng1_acc_list]}', file=f)
            print(f'val_epochs_ng5_acc_list={[float(f"{each:.4f}") for each in val_epochs_ng5_acc_list]}', file=f)
            print(f'val_epochs_ng10_acc_list={[float(f"{each:.4f}") for each in val_epochs_ng10_acc_list]}', file=f)
            print(f'val_epochs_ng20_acc_list={[float(f"{each:.4f}") for each in val_epochs_ng20_acc_list]}', file=f)
            print(f'val_epochs_mAP20_list={[float(f"{each:.4f}") for each in val_epochs_mAP20_list]}', file=f)
            print(f'val_epochs_mrr_list={[float(f"{each:.4f}") for each in val_epochs_mrr_list]}', file=f)


if __name__ == '__main__':
    args = parameter_parser()
    # The name of node features in NYC/graph_X.csv
    args.feature1 = 'checkin_cnt'
    args.feature2 = 'poi_catid'
    args.feature3 = 'latitude'
    args.feature4 = 'longitude'
    train(args)
