import scipy.io as sio
import numpy as np
import torch

data_names = {"brain": "human_brain",
              "ESC": "human_ESC",
              "mESC": "mESC169",
              "mBladder": "mBladder141",
              "tc": "time_course135"}


def get_data(data_name, shape, device):
    filename = "/root/SC_DscNet_k/datasets/" + data_names[data_name] + ".mat"

    # 读取数据
    data = sio.loadmat(filename)
    # print(data['fea'][:, :21609].shape)
    # data1 = np.append(np.array(data['fea']), np.zeros(126).reshape((-1, 1)), axis=1)
    features, labels = data['fea'][:, :21609].reshape((-1, 1, shape, shape)), data['label']
    features = np.log10(features + 1)       # single cell 一般会有这个处理

    # 计算缺失矩阵
    ori_matrix = data['fea'][:, :21609]
    dropout_matrix = np.ones_like(ori_matrix)
    dropout_matrix[ori_matrix == 0] = 0
    idx = np.argwhere(np.all(dropout_matrix[..., :] == 0, axis=0))      # 列为全0的数据不具备参考性，直接取0
    dropout_matrix[:, idx] = 1
    dropout_matrix = dropout_matrix.reshape((-1, 1, shape, shape))

    # 数据简单处理
    features = torch.from_numpy(features).float().to(device)
    labels = np.squeeze(labels - np.min(labels))
    dropout_matrix = torch.from_numpy(dropout_matrix).float().to(device)

    return features, labels, dropout_matrix


if __name__ == "__main__":
    for name in data_names.keys():

        from config import get_config

        cfg = get_config(name)
        x, y, dropout = get_data(name, cfg.shape, 'cpu')
        print(name)
        print(x.shape)
        print(np.unique(y))
        print(len(np.unique(y)))
        print(dropout.shape)
