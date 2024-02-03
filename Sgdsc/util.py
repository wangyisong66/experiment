import os
import random

import torch
import numpy as np
import scipy.io as sio
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn.preprocessing import normalize


def check_path():
    if os.path.exists("results") is False:
        os.makedirs('results')


def load_model(model, name, state):
    state_dict = torch.load(state + '/%s_model.pkl' % name, map_location='cpu')
    model.load_state_dict(state_dict)


def save_mode(model, name):
    torch.save(model.state_dict(), 'results/%s_model.pkl' % name)


def cal_laplacian(coe):
    # 计算laplacian
    d_vec = np.sum(coe, axis=1)
    d_vec = 1 / np.sqrt(d_vec)
    d_matrix = np.diag(d_vec)

    return np.eye(coe.shape[0]) - np.matmul(np.matmul(d_matrix, coe), d_matrix)


def saveInfo(info, name):
    sio.savemat("./results" + name + ".mat", info)


def consine_sim(fea1, fea2):
    cos_sim = cosine_similarity(fea1, fea2)

    return cos_sim


# Sparse the affinity matrix by k-nearst neighbor
# Input: W: NxN affinity matrix,K: number of neighbors
# Output:W_knn: NxN sparsed affinity matrix
def knnSparse(w, k):
    n = w.shape[0]
    idx_knn = np.argsort(-w, 1)  # 降序行排列取索引
    W_knn = np.zeros((n, n))
    for i in range(n):
        W_knn[i, idx_knn[i, 0:k]] = w[i, idx_knn[i, 0:k]]
    W_knn = (W_knn + W_knn.T) / 2

    return W_knn


def getKnn(w, k, need_knn=True, shrink=1):
    w = 1 / 2 * (np.abs(w) + np.abs(w.T))

    # Pre-processing of affinity matrix W
    d = np.sum(w, 1) + np.spacing(1)
    D = np.diag(d)
    w = w - np.diag(np.diag(w)) + D

    # Normalization:W = W ./ repmat(sum(W, 2)+eps, 1, n)
    d = np.sum(w, 1) + np.spacing(1)
    d = 1 / np.sqrt(d)
    D = np.diag(d)
    w = np.matmul(np.matmul(D, w), D)  # nomalize the row of W

    if need_knn:
        return shrink * knnSparse(w, k)
    else:
        return shrink * w


# %%
# Diffusion Process
def IterativeDiffusionTPGKNN(w, k, need_knn=False, shrink=1):
    s = getKnn(w, k, need_knn, shrink)

    WW = s

    maxIter = 50  # 最大迭代次数！！
    epsilon = 1e-2

    for t in range(maxIter):
        temp = np.dot((np.dot(s, WW)), s.T) + np.eye(max(WW.shape))  # 构造高阶张量图
        if np.linalg.norm(temp - WW, ord='fro') < epsilon:  # 两次迭代结果的fro范数差别不大为止
            print('iter_time:%02d' % t)
            break
        WW = temp

    if need_knn:
        WW = knnSparse(WW, k)

    return WW, knnSparse(s, k)


def normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized_matrix = matrix / norms
    return normalized_matrix


def set_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.random.manual_seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num)
        torch.backends.cudnn.deterministic = True


def plot_embedding_2d(data, labels):
    # 标签到颜色的映射
    color_map = {1: 'b', 2: 'c', 3: 'g', 4: 'k', 5: 'm',
                 6: 'coral', 7: 'darkgreen', 8: 'deeppink', 9: 'lavender', 10: 'lime',
                 11: 'olive', 12: 'sienna', 13: 'steelblue', 14: 'yellowgreen', 15: 'whitesmoke',
                 16: 'aliceblue', 17: 'aquamarine', 18: 'black', 19: 'cadetblue', 20: 'cornflowerblue'}

    # 绘制散点图
    for i in range(len(data)):
        x, y = data[i]
        label = labels[i]
        plt.scatter(x, y, c=color_map.get(label, 'black'))

    # 添加图例
    plt.legend()
    plt.show()


def plot_tsne(x, labels, plot=True):
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_tsne = tsne.fit_transform(x)

    if plot:
        plot_embedding_2d(x_tsne, labels)

    return x_tsne


def calculate_cosine_similarity(x1, x2, zero_diag=False):
    x1_ = normalize(x1, axis=1)
    x2_ = normalize(x2, axis=1)
    similarity = np.matmul(x1_, x2_.T)

    if zero_diag:
        similarity = similarity - np.eye(similarity.shape[0], dtype=np.float32)

    return similarity


def knn(matrix, k=10, largest=True):
    # 取出每一行前k个最大值的索引
    _, indices = torch.topk(matrix, k=k, dim=1, largest=largest, sorted=True)

    # 将其他元素置零
    mask = torch.zeros_like(matrix)
    mask.scatter_(1, indices, 1)
    # matrix *= mask
    # matrix = matrix - torch.diag(torch.diag(matrix))
    # m1 = matrix.cpu().numpy()
    # 返回保留前k个元素后的矩阵
    return mask
