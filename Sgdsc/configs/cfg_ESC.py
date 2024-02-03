class ConfigEsc:
    # 数据集参数
    num_cluster = 7
    num_sample = 1018
    shape = 137

    # 网络参数
    channels = [1, 3, 5, 7]
    kernels = [3, 3, 3]
    padding = [1, 1, 2]

    # raw 参数
    raw_epochs = 1200
    raw_lr = 0.001

    # 预训练参数
    pre_epochs = 1800
    pre_lr = 0.001

    # 训练参数  600epoch
    epochs = 500
    lr = 0.001
    weight_coe = 2
    weight_self_exp = 10
    weight_coe2 = 1

    # post clustering parameters
    alpha = 0.06  # threshold of C
    dim_subspace = 12  # dimension of each subspace
    ro = 3  #

    # 其它参数
    comment64 = True
    show_freq = 1
