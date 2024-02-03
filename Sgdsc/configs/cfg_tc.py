class ConfigTC:
    # 数据集参数
    num_cluster = 6
    num_sample = 758
    shape = 135

    # 网络参数
    channels = [1, 3, 5, 7]
    kernels = [3, 3, 3]
    padding = [2, 1, 2]

    # raw 参数
    raw_epochs = 1200
    raw_lr = 0.001

    # 预训练参数
    pre_epochs = 600
    pre_lr = 0.001

    # 训练参数  600epoch
    epochs = 100
    lr = 0.001
    weight_coe = 2
    weight_self_exp = 0.2

    # post clustering parameters
    alpha = 0.14  # threshold of C
    dim_subspace = 15  # dimension of each subspace
    ro = 3  #

    # 其它参数
    comment64 = True
    show_freq = 1
