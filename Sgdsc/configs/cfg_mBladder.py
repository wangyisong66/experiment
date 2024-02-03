class ConfigMBladder:
    # 数据集参数，输入特征维度22085
    num_cluster = 16
    num_sample = 2746
    shape = 141

    # 网络参数
    channels = [1, 3, 5, 7]
    kernels = [3, 3, 3]
    padding = [1, 2, 1]

    # raw 参数
    raw_epochs = 1200
    raw_lr = 0.001

    # 预训练参数
    pre_epochs = 600
    pre_lr = 0.001

    # 训练参数  600epoch
    epochs = 90
    lr = 0.001
    weight_coe = 2.0
    weight_self_exp = 1
    weight_coe2 = 120

    # post clustering parameters
    alpha = 0.12  # threshold of C
    dim_subspace = 13  # dimension of each subspace
    ro = 8  #

    # 其它参数
    comment64 = True
    show_freq = 1
