import numpy as np
import pandas as pd
import torch

def load_adj_data(path,dtype = np.float32):
    #path是所有data所处的文件夹，进到函数后加文件名
    adj_path = path + 'adj.csv'
    adj_df = pd.read_csv(adj_path,header=None)
    adj_array = np.array(adj_df,dtype=dtype)
    #[num_nodes, num_nodes]
    return adj_array

def load_speed_data(path,dtype = np.float32):
    speed_path = path + "speed.csv"
    speed_df = pd.read_csv(speed_path,header=None)
    speed_array = np.array(speed_df,dtype=dtype)
    #[num_noeds,time_len]
    return speed_array

#以下两个是环境特征
def load_poi_feat_data(path):
    poi_feat_path = path + "poi_feat.csv"
    poi_feat_df = pd.read_csv(poi_feat_path)
    poi_feat_array = np.array(poi_feat_df)
    # [num_nodes,poi_feat_dim=14] #sz
    return poi_feat_array

def load_structure_feat_data(path):
    structure_feat_path = path + "structure_feat.csv"
    structure_feat_df = pd.read_csv(structure_feat_path)
    structure_feat_array = np.array(structure_feat_df)
    # [num_nodes,structure_feat_dim=6]
    return structure_feat_array

#预先处理好的embedding，包括node2vec embedding; time_of_day embedding; day of week embedding；envs_embedding
def load_embedding(path,embedding_type):
    file_path = path + "{}.npy".format(embedding_type)
    embedding_data = np.load(file_path)
    return embedding_data
def load_missing_mask(path,miss_type,miss_rate,index_id,dtype = np.int16):
    missing_mask_path = path + "mask/missing_mask_{}{}_idx{}.csv".format(miss_type,miss_rate,index_id)
    missing_mask_df = pd.read_csv(missing_mask_path,header=None)
    missing_mask_array = np.array(missing_mask_df,dtype=dtype)
    #[num_nodes,time_len]
    return missing_mask_array

def drop_zero_data(speed_data,seq_len,count_threshold):
    """
    #把某些时刻原始缺失率很大的样本排除掉
    :param speed_data:
    :param seq_len: 输入序列长度
    :param count_threshold:阈值，如果在某一时刻原始缺失路段的数量超过这个阈值，则排除掉该时刻的样本
    :return:
    """
    df = pd.DataFrame(speed_data)
    drop_record = []
    #seq_len = 6  # 用前后1小时插值当前时刻
    zero_count = (df == 0).sum(axis=0)
    for i, count in enumerate(zero_count):
        if count > count_threshold:
            for j in range(0, seq_len + 1):
                drop_record.append(i - j)
                drop_record.append(i + j)
    drop_record = set(drop_record)  # 后面在这个drop_record里面的时刻标号就不参与训练样本生成了
    return drop_record

def generate_dataset(
        speed_data,missing_mask,
        node_embedding,
        time_in_day_embedding,day_in_week_embedding,envs_feat_embedding,
        seq_len,
        model_name,
        count_threshold,
        dataset_name,
        split_ratio=[0.6,0.2,0.2],
        normalize=True
):
    """
    :param split_ratio: 训练集、验证机和测试集划分比例
    :param normalize:数据是否标准化
    :return:
    """
    drop_record = drop_zero_data(speed_data,seq_len,count_threshold)
    time_len = speed_data.shape[1]-len(drop_record)-seq_len*2
    train_size = int(time_len*split_ratio[0])
    val_size = int(time_len*split_ratio[1])
    test_size = time_len-train_size-val_size

    if normalize:
        max_val = np.max(speed_data)
        speed_data = speed_data / max_val
    all_input, all_label, all_mask,all_embed = [],[],[],[]
    for i in range(seq_len, speed_data.shape[1] - seq_len):
        # 在drop_record里包含的时刻都是涉及到几乎所有路段都是缺失的
        if i in drop_record:
            continue
        speed_i = np.copy(speed_data[:, i])
        missing_mask_i = np.copy(missing_mask[:, i])
        observed_mask_i = np.where(missing_mask_i == 0, 1, 0)
        observed_speed_i = np.multiply(speed_i, observed_mask_i)

        node_embed_i = np.copy(node_embedding[:, i, :])
        time_in_day_embed_i = np.copy(time_in_day_embedding[:, i, :])
        day_in_week_embed_i = np.copy(day_in_week_embedding[:, i, :])
        envs_embed_i = np.copy(envs_feat_embedding[:, i, :])

        speed_past_range = np.copy(speed_data[:, i - seq_len:i])
        speed_future_range = np.copy(speed_data[:, i + 1:i + seq_len + 1])

        embed_i = np.concatenate(
            [node_embed_i, time_in_day_embed_i, day_in_week_embed_i, envs_embed_i, speed_past_range,
             speed_future_range], axis=1)  # [N,dim]

        all_input.append(observed_speed_i)
        all_label.append(speed_i)
        all_mask.append(missing_mask_i)
        all_embed.append(embed_i)

    train_input = all_input[:train_size]
    train_label = all_label[:train_size]
    train_mask = all_mask[:train_size]
    train_embed = all_embed[:train_size]

    val_input = all_input[train_size:train_size + val_size]
    val_label = all_label[train_size:train_size + val_size]
    val_mask = all_mask[train_size:train_size + val_size]
    val_embed = all_embed[train_size:train_size + val_size]

    test_input = all_input[train_size + val_size:time_len]
    test_label = all_label[train_size + val_size:time_len]
    test_mask = all_mask[train_size + val_size:time_len]
    test_embed = all_embed[train_size + val_size:time_len]

    return np.array(train_input), np.array(train_label), np.array(train_mask), np.array(train_embed), np.array(
        val_input), np.array(val_label), np.array(val_mask), np.array(val_embed), np.array(test_input), np.array(
        test_label), np.array(test_mask), np.array(test_embed)


def generate_torch_dataset(
        speed_data,missing_mask,node_embedding,time_in_day_embedding,day_in_week_embedding,envs_feat_embedding,seq_len,model_name,count_threshold,dataset_name,split_ratio=[0.6,0.2,0.2],normalize=True
):
    train_input, train_label, train_mask, train_embed, val_input, val_label, val_mask, val_embed, test_input, test_label, test_mask, test_embed = generate_dataset(
        speed_data, missing_mask, node_embedding,time_in_day_embedding,day_in_week_embedding,envs_feat_embedding,seq_len,model_name,count_threshold,dataset_name)

    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_input),torch.FloatTensor(train_label),torch.FloatTensor(train_mask),torch.FloatTensor(train_embed)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(val_input), torch.FloatTensor(val_label), torch.FloatTensor(val_mask),torch.FloatTensor(val_embed)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_input), torch.FloatTensor(test_label), torch.FloatTensor(test_mask),torch.FloatTensor(test_embed)
    )
    return train_dataset, val_dataset,test_dataset

def envs_feat_process(poi_feat_data, structure_feat_data):
    """
    处理环境因子，拼接整合成一个向量
    :param poi_feat_data:
    :param structure_feat_data:
    :return:
    """
    #原来直接合并在一起
    #feat = np.concatenate([poi_feat_data, structure_feat_data], axis=1)  # (23,) dim=23
    #poi_feat [餐饮服务,公司企业,交通设施服务,科教文化服务,汽车服务,生活服务,医疗保健服务,体育休闲服务,政府机构及社会团体,住宿服务,商务住宅,购物服务,金融保险服务,风景名胜]
    #structure_feat [length,centrality,betweenness,closeness,curve,level]
    #poi特征经过tf-idf加权处理，因此这里仅对structure特征进行处理
    #先写成固定的了
    new_structure_feat_data = np.zeros((structure_feat_data.shape[0],9))
    for i in range(structure_feat_data.shape[1]-1):
        new_structure_feat_data[:,i] = (structure_feat_data[:,i] - np.min(structure_feat_data[:,i])) / (np.max(structure_feat_data[:,i]) - np.min(structure_feat_data[:,i]))
    i = structure_feat_data.shape[1]-1
    #等级编码为one-hot向量
    for j in range(structure_feat_data.shape[0]):
        new_structure_feat_data[j,i+structure_feat_data[j,i].astype(int)-1] = 1

    feat = np.concatenate([poi_feat_data, new_structure_feat_data], axis=1)  # (20,) dim=23
    return feat

def load_data(path, index_id,miss_type,miss_rate,model_name,dataset_name):
    path = path + "{}_data/".format(dataset_name)
    if dataset_name == 'metr_la':
        seq_len = 12
        count_threshold = 200
    speed_data = load_speed_data(path)
    max_val = np.max(speed_data)
    missing_mask = load_missing_mask(path,miss_type,miss_rate,index_id)
    adj = load_adj_data(path)
    node_embedding_dim = 16
    node_embedding = load_embedding(path,"node_embedding_{}".format(node_embedding_dim))
    time_in_day_embedding = load_embedding(path,"time_in_day_embedding")
    day_in_week_embedding = load_embedding(path,'day_in_week_embedding')
    envs_feat_embedding = load_embedding(path,"envs_feat_embedding")

    train_dataset, val_dataset, test_dataset = generate_torch_dataset(speed_data, missing_mask, node_embedding,time_in_day_embedding,day_in_week_embedding,envs_feat_embedding,seq_len,model_name,count_threshold,dataset_name)

    return train_dataset, val_dataset, test_dataset, adj, max_val



