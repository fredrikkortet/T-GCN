import numpy as np
import pandas as pd
import torch
from utils.graph_conv_att import get_attention_adj_matrix

def svd_low_rank_approximation(matrix, k=50):
    if k == 0:
        return matrix
    # Perform SVD on the matrix
    u, s, v = np.linalg.svd(matrix)
    
    # Truncate the singular values and matrices to rank k
    s_k = s[:k]
    u_k = u[:, :k]
    v_k = v[:k, :]
    
    
    return np.dot(u_k ,np.dot(np.diag(s_k) , v_k))

def load_features(feat_path,svd , dtype=np.float32):
    if feat_path.endswith(".npz"):
        data = np.load(feat_path)
        data = data.f.data
        df_data = pd.DataFrame(data[:,:,0])
        df_data = df_data.values.tolist()
        feat = np.array(df_data, dtype=dtype)
        feat = svd_low_rank_approximation(feat,svd)
        return feat
    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)
    feat = svd_low_rank_approximation(feat,svd)
    return feat


def load_adjacency_matrix(adj_path,self_attention, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    if self_attention == 1:
        adj=get_attention_adj_matrix(adj)
    return adj


def generate_dataset(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    """
    :param data: feature matrix
    :param seq_len: length of the train data sequence
    :param pre_len: length of the prediction data sequence
    :param time_len: length of the time series in total
    :param split_ratio: proportion of the training set
    :param normalize: scale the data to (0, 1], divide by the maximum value in the data
    :return: train set (X, Y) and test set (X, Y)
    """
    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        max_val = np.max(data)
        data = data / max_val
    train_size = int(time_len * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:time_len]
    train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
    for i in range(len(train_data) - seq_len - pre_len):
        train_X.append(np.array(train_data[i : i + seq_len]))
        train_Y.append(np.array(train_data[i + seq_len : i + seq_len + pre_len]))
    for i in range(len(test_data) - seq_len - pre_len):
        test_X.append(np.array(test_data[i : i + seq_len]))
        test_Y.append(np.array(test_data[i + seq_len : i + seq_len + pre_len]))
    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)


def generate_torch_datasets(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    train_X, train_Y, test_X, test_Y = generate_dataset(
        data,
        seq_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
    )
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
    )
    return train_dataset, test_dataset
