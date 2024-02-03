class ConfigBrain:
    # 数据集参数，输入特征维度22085
    num_cluster = 8
    num_sample = 420
    shape = 147

    # 网络参数
    channels = [1, 3, 5, 7]
    kernels = [3, 3, 3]
    padding = [2, 2, 1]

    # raw 参数
    raw_epochs = 1200
    raw_lr = 0.001

    # 预训练参数
    pre_epochs = 600
    pre_lr = 0.001

    # 训练参数  600epoch
    epochs = 200
    lr = 0.001
    weight_coe = 2.0
    weight_self_exp = 0.2
    weight_coe2 = 80

    # post clustering parameters
    alpha = 0.2  # threshold of C
    dim_subspace = 3  # dimension of each subspace
    ro = 1  #

    # 其它参数
    comment64 = True
    show_freq = 1
