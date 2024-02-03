import torch
from torch import optim
import matplotlib.pyplot as plt
import evaluation
from config import get_config
from evaluate import get_score
from load_data import get_data
from network import DSCNet
from util import load_model, save_mode, plot_tsne, calculate_cosine_similarity
from post_clustering import spectral_clustering


class RunModel:
    def __init__(self, name):
        # get configs
        self.cfg = get_config(name)
        # get device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # get dataloader
        self.features, self.labels, self.dropout = get_data(name, self.cfg.shape, self.device)
        # set name
        self.name = name

    def get_train_param(self):
        cfg = self.cfg
        return cfg.epochs, cfg.weight_coe, cfg.weight_self_exp

    def get_cluster_param(self):
        cfg = self.cfg
        return cfg.num_cluster, cfg.dim_subspace, cfg.alpha, cfg.ro, cfg.comment64, cfg.show_freq

    def train_raw_ae(self):
        # 模型参数
        cfg = self.cfg
        model = DSCNet(num_sample=cfg.num_sample, channels=cfg.channels,
                       padding=cfg.padding, kernels=cfg.kernels).to(self.device)

        # get training param
        lr = cfg.raw_lr
        epochs = cfg.raw_epochs

        optimizer = optim.Adam(model.ae.parameters(), lr=lr)
        x = self.features
        dropout = self.dropout

        # plot t-sne raw
        plot_tsne(x.view(self.cfg.num_sample, -1).detach().cpu().numpy(), self.labels)

        model.train()
        for epoch in range(epochs):
            loss = model.loss_ae(x, dropout)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch: {}, loss: {}".format(epoch, loss))

            if epoch % 100 == 0 and epoch > 0:
                with torch.no_grad():
                    # construct graph from z
                    z, x_recon = model.forward_ae(x)

                    # K-means
                    z = z.reshape(cfg.num_sample, -1).detach().cpu().numpy()
                    scores = evaluation.clustering([z], self.labels)['kmeans']
                    print("\033[2;29m" + str(scores) + "\033[0m")

                    # # # get impute
                    # fea_imp = x_recon * (1 - self.dropout) + x * self.dropout
                    # fea_imp = fea_imp.reshape(cfg.num_sample, -1)
                    # fea_imp = fea_imp.detach().cpu().numpy()
                    # df = pd.DataFrame(fea_imp)
                    # df.to_csv(self.name + str(epoch) + '.csv')

        save_mode(model, self.name + "_raw")

    def train_raw_dsc(self):
        cfg = self.cfg

        # 模型参数
        model = DSCNet(num_sample=cfg.num_sample, channels=cfg.channels, padding=cfg.padding, kernels=cfg.kernels).to(
            self.device)
        load_model(model, self.name + "_raw", 'results')

        model.train()
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        epochs, weight_coe, weight_self_exp = self.get_train_param()
        weight_coe2 = 0
        num_cluster, dim_subspace, alpha, ro, comment64, show_freq = self.get_cluster_param()
        x = self.features
        dropout = self.dropout
        y = self.labels

        graph_init = calculate_cosine_similarity(x, x, False)
        graph_init = torch.from_numpy(graph_init).to(self.device)

        for epoch in range(epochs):
            loss = model.loss_dsc(x, dropout, weight_coe=weight_coe, weight_self_exp=weight_self_exp,
                                  init_graph=graph_init, weight_coe2=weight_coe2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch % 1 == 0 or epoch == epochs - 1) and epoch >= 0:
                coe = model.self_expression.Coefficient.detach().to('cpu').numpy()
                y_pred = spectral_clustering(coe, num_cluster, dim_subspace, alpha, ro, comment64)
                acc, nmi = get_score(y, y_pred)
                print('Epoch %02d: loss=%.4f, acc=%.4f, nmi=%.4f' % (epoch, loss.item() / y_pred.shape[0], acc, nmi))

    def train_imp_ae(self):
        cfg = self.cfg

        # get imp data
        with torch.no_grad():
            model = DSCNet(num_sample=cfg.num_sample, channels=cfg.channels, padding=cfg.padding,
                           kernels=cfg.kernels).to(self.device)
            load_model(model, self.name + "_raw", 'results')
            _, x_recon = model.forward_ae(self.features)
            fea_imp = x_recon * (1 - self.dropout) + self.features * self.dropout

        model = DSCNet(num_sample=cfg.num_sample, channels=cfg.channels,
                       padding=cfg.padding, kernels=cfg.kernels).to(self.device)

        # get training param
        lr = cfg.raw_lr
        epochs = cfg.raw_epochs

        optimizer = optim.Adam(model.ae.parameters(), lr=lr)
        x = fea_imp
        dropout = torch.ones_like(self.dropout).to(self.device)
        model.train()
        for epoch in range(epochs):
            loss = model.loss_ae(x, dropout)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch: {}, loss: {}".format(epoch, loss))

            if epoch % 100 == 0 and epoch > 0:
                with torch.no_grad():
                    # construct graph from z
                    z, x_recon = model.forward_ae(x)

                    # K-means
                    z = z.reshape(cfg.num_sample, -1).detach().cpu().numpy()
                    scores = evaluation.clustering([z], self.labels)['kmeans']
                    print("\033[2;29m" + str(scores) + "\033[0m")

        save_mode(model, self.name + "_imp")

    def train_imp_dsc(self):
        # 模型参数
        cfg = self.cfg

        # get imp data
        with torch.no_grad():
            model = DSCNet(num_sample=cfg.num_sample, channels=cfg.channels, padding=cfg.padding,
                           kernels=cfg.kernels).to(self.device)
            load_model(model, self.name + "_raw", 'results')
            _, x_recon = model.forward_ae(self.features)
            fea_imp = x_recon * (1 - self.dropout) + self.features * self.dropout

            # plot t-sne raw
            # plot_tsne(fea_imp.view(self.cfg.num_sample, -1).detach().cpu().numpy(), self.labels)

        # init graph
        x_tmp = fea_imp.view(self.cfg.num_sample, -1).detach().cpu().numpy()
        graph_init = calculate_cosine_similarity(x_tmp, x_tmp, False)
        graph_init = torch.from_numpy(graph_init).to(self.device)

        # load pretrain data
        model = DSCNet(num_sample=cfg.num_sample, channels=cfg.channels, padding=cfg.padding,
                       kernels=cfg.kernels).to(self.device)
        load_model(model, self.name + "_imp", 'results')

        model.train()
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        epochs, weight_coe, weight_self_exp = self.get_train_param()
        weight_coe2 = cfg.weight_coe2
        num_cluster, dim_subspace, alpha, ro, comment64, show_freq = self.get_cluster_param()
        # model.set_coe(graph_init)
        losses = []
        x = fea_imp
        dropout = torch.ones_like(self.dropout).to(self.device)
        y = self.labels
        for epoch in range(epochs):
            loss = model.loss_dsc(x, dropout, y, weight_coe=weight_coe, weight_self_exp=weight_self_exp,
                                  init_graph=graph_init, weight_coe2=weight_coe2)
            losses.append(loss.item() / x.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch % 10 == 0 or epoch == epochs - 1) and epoch >= 0:
                coe = model.self_expression.Coefficient.detach().to('cpu').numpy()
                y_pred = spectral_clustering(coe, num_cluster, dim_subspace, alpha, ro, comment64)
                acc, nmi = get_score(y, y_pred)

                print('Epoch %02d: loss=%.4f, acc=%.4f, nmi=%.4f' % (epoch, loss.item() / y_pred.shape[0], acc, nmi))
        plt.figure()
        plt.plot(losses)
        # 添加标题和标签
        plt.title('LOSS')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        # 添加图例
        plt.legend()
        plt.savefig(f'/root/SC_DscNet_k/result_{self.name}.jpg')
        # 显示图形
        plt.show()

    def get_imputation(self):
        cfg = self.cfg
        model = DSCNet(num_sample=cfg.num_sample, channels=cfg.channels,
                       padding=cfg.padding, kernels=cfg.kernels).to(self.device)
        load_model(model, self.name + "_raw", 'results')

        with torch.no_grad():
            fea_imp = model.ae(self.features) * (1 - self.dropout) + self.features * self.dropout

        return fea_imp.reshape(cfg.num_sample, -1)


if __name__ == "__main__":
    print("a")
    a = RunModel("orl")
    print(len(a.features))
    print(a.features[0].shape, a.features[1].shape, a.features[0].dtype)
