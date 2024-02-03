import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class ConvAE(nn.Module):
    def __init__(self, channels, kernels, padding):
        super(ConvAE, self).__init__()
        assert isinstance(channels, list) and isinstance(kernels, list)
        self.encoder = nn.Sequential()
        # o = [ (i + 2p - k)/s ] +1
        for i in range(1, len(channels)):
            #  Each layer will divide the size of feature map by 2
            self.encoder.add_module('conv%d' % i,
                                    nn.Conv2d(channels[i - 1], channels[i], kernel_size=kernels[i - 1], stride=(2, 2),
                                              padding=(padding[i-1], padding[i-1])))
            self.encoder.add_module('relu%d' % i, nn.ReLU(True))

        self.decoder = nn.Sequential()
        # i' = [ (o' + 2p - k)/s ] +1
        channels = list(reversed(channels))
        kernels = list(reversed(kernels))
        padding = list(reversed(padding))
        for i in range(len(channels) - 1):
            # Each layer will double the size of feature map
            self.decoder.add_module('deconv%d' % (i + 1),
                                    nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernels[i],
                                                       stride=(2, 2), padding=(padding[i], padding[i])))
            self.decoder.add_module('relud%d' % i, nn.ReLU(True))

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y


class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)

    def forward(self, x):  # shape=[n, d]
        y = torch.matmul(self.Coefficient, x)
        return y


class DSCNet(nn.Module):
    def __init__(self, channels, kernels, padding, num_sample):
        super(DSCNet, self).__init__()
        self.n = num_sample
        self.ae = ConvAE(channels, kernels, padding)
        self.self_expression = SelfExpression(self.n)

    def set_coe(self, graph_init):
        with torch.no_grad():
            self.self_expression.Coefficient.data = graph_init

    def get_coe(self):
        with torch.no_grad():
            coe = self.self_expression.Coefficient.detach().clone()

        return coe

    def get_imputation(self, x):
        with torch.no_grad():
            shape = x.shape
            x = x.view(self.n, -1)
            x_imputation = self.self_expression(x).reshape(shape)

        return x_imputation

    def forward_ae(self, x):
        z = self.ae.encoder(x)
        x_recon = self.ae.decoder(z)

        return z, x_recon

    def forward(self, x):  # shape=[n, c, w, h]
        z = self.ae.encoder(x)

        # self expression layer, reshape to vectors, multiply Coefficient, then reshape back
        shape = z.shape
        z = z.view(self.n, -1)  # shape=[n, d]
        z_recon = self.self_expression(z)  # shape=[n, d]
        z_recon_reshape = z_recon.view(shape)

        x_recon = self.ae.decoder(z_recon_reshape)  # shape=[n, c, w, h]
        return x_recon, z, z_recon

    def loss_ae(self, x, dropout):
        _, x_recon = self.forward_ae(x)

        return F.mse_loss(x_recon * dropout, x, reduction='sum')

    def loss_dsc(self, x, dropout, y, weight_coe, weight_self_exp, init_graph, weight_coe2):
        x_recon, z, z_recon = self.forward(x)
        # 使用 t-SNE 进行降维
        tsne = TSNE(n_components=2, init='pca', random_state=42)
        X_tsne = tsne.fit_transform(z.detach().numpy())
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        plt.figure(figsize=(8, 8))
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
                     fontdict={'weight': 'bold', 'size': 9})

        # 可视化 t-SNE 图像
        # plt.scatter(z_tsne[:, 0], z_tsne[:, 1])
        plt.title('t-SNE Visualization')
        plt.savefig('/root/SC_DscNet_k/result_tsne.jpg')
        plt.show()
        # recon loss
        loss_ae = F.mse_loss(x_recon * dropout, x, reduction='sum')

        # norm loss
        loss_coe = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        loss_coe2 = torch.sum(torch.abs(self.self_expression.Coefficient * (1 - init_graph)))

        # self express loss
        loss_self_exp = F.mse_loss(z_recon, z, reduction='sum')

        loss = loss_ae + weight_coe * loss_coe + weight_self_exp * loss_self_exp + weight_coe2 * loss_coe2

        return loss


if __name__ == "__main__":
    from config import get_config
    import numpy as np

    cfg = get_config("brain")
    net = DSCNet(cfg.channels, cfg.kernels, cfg.padding, cfg.num_sample)
    print(net)

    fea_len = 149
    data = torch.from_numpy(np.ones((fea_len * fea_len * cfg.num_sample)).reshape((-1, 1, fea_len, fea_len))).float()
    x_recon1, z1, z_recon1 = net(data)
    loss3 = net.loss_fn(data, x_recon1, z1, z_recon1, 1, 1)
    print(z1.shape)
    print(x_recon1.shape)
    print(loss3)
