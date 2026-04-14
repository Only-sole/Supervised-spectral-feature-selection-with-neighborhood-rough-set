"""
@author: Only
"""

import numpy as np
from sklearn.metrics import pairwise


def similar(dist, t=1.0):
    '''
    返回相似度
    '''
    return np.exp(-(dist / t))


def euc_dis(X):

    return np.square(pairwise.pairwise_distances(X, metric='euclidean'))



def recent_heterogeneous(X, y, n, num_class, dis):
    '''
    全局最近异类点
    '''
    ans = np.zeros(n, dtype=np.int16)  # 存放最近异类点的序号
    label_sub = []  # 存各类别标签样本序号，列表嵌套[[],[]]
    for i in range(len(num_class)):
        label_sub.append(np.where(y == num_class[i])[0].tolist())

    for i in range(n):
        dis[i, label_sub[int(y[i])]] = np.inf
        ans[i] = np.argmin(dis[i])

    return ans


def delta_neighborhood(data, y, n, hete_recent, delta, beta):
    """
    计算单个特征的邻域
    :param data: 样本特征矩阵，形状为(n, m)，其中n为样本个数，m为特征维度
    :param y: 样本标签矢量，形状为(n,)，其中n为样本个数
    :param n: 样本个数
    :param hete_recent: 每个样本的最近异类点序号
    :param delta: 邻域定义参数
    :param beta: 交集占比阈值
    :return: 邻域矩阵，形状为(n, n)
    """
    # 计算相似度矩阵
    if len(data.shape) == 1:
        similarity = np.exp(-np.square(pairwise.euclidean_distances(data.reshape(-1,1)))/ (2 * np.var(data)))
    else:
        similarity = np.exp(-np.square(pairwise.euclidean_distances(data)) / (2 * np.var(data)))

    # 确定每个样本的邻域
    neighbor = np.zeros((n, n))
    neighbor[similarity >= delta] = 1

    # 判断最近异类点与本身邻域的交集
    for i in range(n):
        # 交集太多
        intersection = np.minimum(neighbor[i], neighbor[hete_recent[i]])
        if np.sum(intersection) / np.sum(neighbor[i]) > beta:
            # 是交集且邻域内是异类
            neighbor[i, (intersection == 1) & (y != y[i])] = 0

    return neighbor


def cal_UD(X, y):
    n, m = X.shape
    # 生成标签矩阵
    UD = [np.where(y == ii, 1, 0) for ii in np.unique(y)]

    return UD


def cal_X_condi_centro(X, y, UD, hete_recent, delta, beta):
    n, m = X.shape
    ans = np.zeros([n, m])
    for i in range(m):
        neibor = delta_neighborhood(X[:, i], y, n, hete_recent, delta, beta)
        for j in range(n):
            count = 0
            tmp2 = np.sum(neibor[j, :])
            for k, kk in enumerate(UD):
                tmp1 = np.sum(np.minimum(neibor[j, :], kk))
                # 防止对数内出现0，导致的空值nan
                if tmp1 != 0:
                    count -= (tmp1 / n) * np.log2(tmp1 / tmp2)               

            ans[j, i] = count + y[j]

    return ans



def main(X, y, gama, lambada, K, maxIter, d_dim, W, delta=0.8, beta=0.8, t=1):
    '''
    
    '''
    ttmmpp = np.unique(y)
    for i in range(len(ttmmpp)):
        y[y == ttmmpp[i]] = i

    n, m = X.shape
    UD = cal_UD(X, y)
    dis = euc_dis(X)   
    if type(dis[0, 0]) is np.int32 or type(dis[0, 0]) is np.int64:
        dis = dis.astype(np.float32)
    num_class = np.unique(y)
    # 最近异类点
    hete_recent = recent_heterogeneous(X, y, n, num_class, dis)
    X_new = cal_X_condi_centro(X, y, UD, hete_recent, delta, beta)
    # X_new = X
    XT = X_new.T

    S = W.copy()
    P = np.ones((m, d_dim)) # 降至d维
    np.fill_diagonal(S, 0)
    S_hat = (S.T + S) / 2
    D = np.diag(np.sum(S_hat, axis=1))
    LS = D - S_hat
    XTLSX = np.dot(np.dot(XT, LS), X_new)
    count = np.zeros(maxIter)

    for i in range(maxIter):
        
        # ------- update P --------
        Q = np.eye(m)
        for j in range(1):
            eig_val, eig_vec = np.linalg.eigh(XTLSX + gama * Q)
            
            sort_index_ = np.argsort(np.abs(eig_val))  
            eig_val = eig_val[sort_index_]
            start = 0
            while start < m and eig_val[start] < 1e-6:
                 start += 1
           
            if m - start < d_dim:
                sort_index_ = sort_index_[-d_dim : ]
                P = eig_vec[:, sort_index_]
                Q = np.diag(1 / (2 * np.sqrt(np.diag(np.dot(P, P.T)) + np.spacing(1))))               
            else:
                sort_index_ = sort_index_[start : start + d_dim]
                P = eig_vec[:, sort_index_]
                Q = np.diag(1 / (2 * np.sqrt(np.diag(np.dot(P, P.T)) + np.spacing(1))))
                
            
        # ------- update S --------
        PTXT = P.T @ XT
        H = np.square(pairwise.euclidean_distances(PTXT.T))
        Z = W - H/(2*lambada)
        
        for j in range(n):
            ft = 1
            indx = np.delete(np.arange(n),j)
            v = Z[j, indx]
            v0 = v - np.mean(v) + 1/n
            vmin = np.min(v0)
            if vmin < 0:
                f = 1
                tmp = 0
                while np.abs(f) > 1E-10:
                    v1 = v0 - tmp
                    posidx = v1>0
                    npos = np.sum(posidx)
                    g = -npos
                    f = np.sum(v1[posidx]) - 1
                    tmp = tmp - f/g
                    ft += 1
                    if ft > 100:
                        S[j, indx] = np.maximum(v1, 0)
                        break
                S[j, indx] = np.maximum(v1, 0)
            else:
                S[j, indx] = v0


        count[i] = count[i] + np.sum(H*S) + gama * np.sum(np.linalg.norm(P, axis=1)) + lambada * np.square(np.linalg.norm(W - S, 'fro'))
        S_hat = (S.T + S) / 2
        D = np.diag(np.sum(S_hat, axis=1))
        LS = D - S_hat
        XTLSX = np.dot(np.dot(XT, LS), X_new)
        if i > 0 and np.abs(count[i] - count[i - 1]) < 1e-6:
            break

    return P, S, X_new, count

