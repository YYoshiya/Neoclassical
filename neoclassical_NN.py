import time
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ValueNetwork(nn.Module):
    def __init__(self, hidden_size=64):
        super(ValueNetwork, self).__init__()
        self.dense1 = nn.Linear(1, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dense3 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leakyreLU = nn.LeakyReLU()

    def forward(self, x):
        x = self.dense1(x)
        x = self.leakyreLU(x)
        x = self.dense2(x)
        x = self.leakyreLU(x)
        x = self.dense3(x)
        x = self.leakyreLU(x)
        x = self.output(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.dense1 = nn.Linear(1, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dense3 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leakyreLU = nn.LeakyReLU()


    def forward(self, x):
        x = self.dense1(x)
        x = self.leakyreLU(x)
        x = self.dense2(x)
        x = self.leakyreLU(x)
        x = self.dense3(x)
        x = self.leakyreLU(x)
        x = self.output(x)
        return x
    
def pretrain(network, optimizer, data_loader, grid_k, epochs=1000, lr=0.01):
    optimizer = optim.Adam(network.parameters(), lr=lr)
    criterion = nn.MSELoss()  # 平均二乗誤差を損失関数として使用

    for epoch in range(epochs):
        total_loss = 0
        for data in data_loader:
            inputs = data.to(device)  # dataloaderからのデータがGPUに転送
            targets = inputs  # プレトレーニングではgrid_k自体をターゲットとする

            optimizer.zero_grad()  # 勾配を初期化
            outputs = network(inputs)  # ネットワークの出力を計算
            loss = criterion(outputs, targets)  # 損失を計算
            loss.backward()  # 勾配を計算
            optimizer.step()  # オプティマイザを使ってパラメータを更新

            total_loss += loss.item()

    # 最終的な出力をプロットする
    print("Final output plot after pretraining:")

    # grid_kを入力に使って出力をプロット
    inputs = torch.tensor(grid_k, dtype=torch.float32).unsqueeze(1).to(device)  # grid_kをテンソルに変換しGPUに転送
    outputs = network(inputs).detach().cpu().numpy()  # ネットワークの出力を計算

    # プロット
    plt.plot(grid_k, outputs, label='Pretrained Output')
    plt.plot(grid_k, grid_k, linestyle='--', label='Grid_k (Target)')
    plt.title('Network Output After Pretraining')
    plt.xlabel('Input (grid_k)')
    plt.ylabel('Output')
    plt.legend()
    plt.show()
    
class CapitalDataset(Dataset):
    def __init__(self, data):
        super(CapitalDataset, self).__init__()
        self.data = torch.tensor(data, dtype=torch.float32).view(-1,1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def utility(c, sigma):
    eps = torch.tensor(1e-8).to(device)

    if sigma == 1:
        return torch.log(torch.max(c, eps))
    else:
        return (torch.max(c, eps)**(1-sigma) - 1) / (1 - sigma)

class Neoclassical_NN:
    def __init__(self, plott=1, transition=1):
        self.plott = plott  # select 1 to make plots
        self.transition = transition  # select 1 to do transition
        self.value = ValueNetwork().to(device)  # モデルをGPUに転送
        self.policy = PolicyNetwork().to(device)  # モデルをGPUに転送
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=0.0001)
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=0.0001)

        # 学習率減衰の設定 (0.01 -> 0.00001)
        gamma = (0.00001 / 0.01) ** (1 / 1000)
        self.setup_parameters()
        self.setup_grid()

        # Dataset and dataloader creation
        self.batch_size = 128
        self.dataset = CapitalDataset(self.grid_k)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.dataloader_policy = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_dataloader = DataLoader(self.dataset, batch_size=self.Nk, shuffle=False)

        self.params = self.sigma, self.beta, self.delta, self.alpha, self.grid_k, self.tol, self.epoch

        if self.plott != 1 and self.plott != 0:
            raise Exception("Plot option incorrectly entered: Choose either 1 or 0.")

        if self.transition != 1 and self.transition != 0:
            raise Exception("Transition option incorrectly entered: Choose either 1 or 0.")

    def setup_parameters(self):
        self.sigma = 2  # CRRA coefficient
        self.beta = 0.95  # discount factor
        self.delta = 0.1  # depreciation rate
        self.alpha = 1 / 3  # Cobb-Douglas coefficient

        self.k_ss = (self.alpha / (1 / self.beta - (1 - self.delta))) ** (1 / (1 - self.alpha))
        self.c_ss = self.k_ss ** self.alpha - self.delta * self.k_ss
        self.i_ss = self.delta * self.k_ss
        self.y_ss = self.k_ss ** self.alpha

        self.tol = 1e-8
        self.maxit = 2000
        self.epoch = 1000

        self.Nk = 128*5
        self.dev = 0.9
        self.k_min = (1 - self.dev) * self.k_ss
        self.k_max = (1 + self.dev) * self.k_ss

        self.sim_T = 75
        self.perc = 0.5
        
        self.epoch_v = 20
        self.epoch_p = 20


    def setup_grid(self):
        self.grid_k = np.linspace(self.k_min, self.k_max, self.Nk)

    def perfect_foresight_transition(self):
        trans_k = torch.zeros(self.sim_T + 1).to(device)
        trans_k[0] = self.perc * self.k_ss
        trans_cons = torch.zeros(self.sim_T).to(device)
        for t in range(self.sim_T):
            trans_cons[t] = self.policy(trans_k[t].view(-1, 1)).squeeze(1)
            trans_k[t + 1] = trans_k[t] ** self.alpha + (1 - self.delta) * trans_k[t] - trans_cons[t]
        trans_output = trans_k[0:-1] ** self.alpha
        trans_inv = trans_k[1:] - (1 - self.delta) * trans_k[0:-1]

        return trans_k.detach().cpu().numpy(), trans_cons.detach().cpu().numpy(), trans_output.detach().cpu().numpy(), trans_inv.detach().cpu().numpy()

    def pretrain_models(self, pretrain_epochs=100):
        print("Pretraining Value Network...")
        pretrain(self.value, self.optimizer_value, self.dataloader, self.grid_k, epochs=pretrain_epochs)

        print("Pretraining Policy Network...")
        pretrain(self.policy, self.optimizer_policy, self.dataloader, self.grid_k, epochs=pretrain_epochs)


    
    def value_train(self, data):
        loss = self.value_loss(data)
        self.optimizer_value.zero_grad()
        loss.backward()
        self.optimizer_value.step()
    
    def value_loss(self, data):
        max_c = data.squeeze()**self.alpha + (1-self.delta)*data.squeeze()
        c = self.policy(data).squeeze(1)
        c = torch.clamp(c, min=torch.tensor(0).to(device), max=max_c)
        u = utility(c, self.sigma)
        k_next = max_c - c
        k_next = k_next.unsqueeze(1)
        bellman = u + self.beta * self.value(k_next).squeeze(1)
        loss = nn.MSELoss()(self.value(data).squeeze(1), bellman.detach())
        return loss


    def policy_train(self, data):
        loss = self.policy_loss(data)
        self.optimizer_policy.zero_grad()
        loss.backward()
        self.optimizer_policy.step()

        
    def policy_loss(self, data):
        max_c = data.squeeze()**self.alpha + (1-self.delta)*data.squeeze()
        c = self.policy(data).squeeze(1)
        c = torch.clamp(c, min=torch.tensor(0).to(device), max=max_c)
        u = utility(c, self.sigma)
        k_next = max_c - c
        k_next = k_next.unsqueeze(1)
        bellman = u + self.beta * self.value(k_next).squeeze(1)
        loss = -torch.mean(bellman)
        return loss
        
    def vfi_det(self):
        policy_count = 0
        valid_loss = torch.zeros((2, self.epoch))
        for t in range(self.epoch):
              # ロスの合計
            # サンプル数をカウント
            
            for v in range(self.epoch_v):
                for k in self.dataloader:
                    k = k.to(device)  # GPUにデータ転送
                    self.value_train(k)
            for p in range(self.epoch_p):
                for k in self.dataloader_policy:
                    k = k.to(device)
                    self.policy_train(k)
            # 20エポックごとにvalueのロスをプリント
            
            with torch.no_grad():
                for k in self.valid_dataloader:
                    k = k.to(device)
                    value_loss = self.value_loss(k)
                    policy_loss = self.policy_loss(k)
                valid_loss[0, t] = value_loss
                valid_loss[1, t] = -policy_loss
                if t > 0:
                    loss_diff_v = torch.abs(valid_loss[0, t] - valid_loss[0, t - 1])
                    loss_diff_p = torch.abs(valid_loss[1, t] - valid_loss[1, t - 1])
                    print(f"Epoch {t + 1}, loss_diff_v: {loss_diff_v:.8f}, value_loss: {value_loss},loss_diff_p: {loss_diff_p:.8f}")
                    if (value_loss < 1e-5 and loss_diff_v < 1e-6 ) or t == 100:
                        break
                #print(f"Epoch {t + 1}, loss_diff: {loss_diff:.8f}")
        grid_k = torch.tensor(self.grid_k, dtype=torch.float32).view(-1, 1).to(device)
        consumption = self.policy(grid_k).squeeze(1).detach().cpu().numpy()
        return consumption, t

    def solve_model(self):
        t0 = time.time()

        # Pretrain models
        self.pretrain_models(pretrain_epochs=20)

        print("\nSolving social planner problem...")
        self.pol_cons, self.t = self.vfi_det()

        if self.t < self.maxit - 1:
            print(f"Converged in {self.t} iterations.")
        else:
            print("Did not converge.")

        t1 = time.time()
        print(f"Time: {t1 - t0:.2f} seconds")

        if self.transition:
            print("\nSteady State Transition...")
            self.trans_k, self.trans_cons, self.trans_output, self.trans_inv = self.perfect_foresight_transition()

            t2 = time.time()
            print(f'Transition iteration time elapsed: {t2 - t1:.2f} seconds')

        if self.plott:
            print('\nPlotting...')
            plt.plot(self.grid_k, self.value(torch.tensor(self.grid_k, dtype=torch.float32).view(-1, 1).to(device)).detach().cpu().numpy())
            plt.title('Value Function')
            plt.xlabel('Capital Stock')
            plt.show()

            plt.plot(self.grid_k, self.policy(torch.tensor(self.grid_k, dtype=torch.float32).view(-1, 1).to(device)).detach().cpu().numpy())
            plt.title('Next Period Capital Stock Policy Function')
            plt.xlabel('Capital Stock')
            plt.plot([self.k_min, self.k_max], [self.k_min, self.k_max], linestyle=':')
            plt.legend(['Policy Function', '45 Degree Line'])
            plt.show()

            plt.plot(self.grid_k, self.pol_cons)
            plt.title('Consumption Policy Function')
            plt.xlabel('Capital Stock')
            plt.show()

            if self.transition:
                plt.plot(np.arange(self.sim_T), self.trans_k[:-1])
                plt.plot(np.arange(self.sim_T), self.k_ss * np.ones(self.sim_T), linestyle='--')
                plt.title('Transition Dynamics: Capital Stock')
                plt.xlabel('Time')
                plt.show()

                plt.plot(np.arange(self.sim_T), self.trans_cons)
                plt.plot(np.arange(self.sim_T), self.c_ss * np.ones(self.sim_T), linestyle='--')
                plt.title('Transition Dynamics: Consumption')
                plt.xlabel('Time')
                plt.show()

                plt.plot(np.arange(self.sim_T), self.trans_output)
                plt.plot(np.arange(self.sim_T), self.y_ss * np.ones(self.sim_T), linestyle='--')
                plt.title('Transition Dynamics: Output')
                plt.xlabel('Time')
                plt.show()

                plt.plot(np.arange(self.sim_T), self.trans_inv)
                plt.plot(np.arange(self.sim_T), self.i_ss * np.ones(self.sim_T), linestyle='--')
                plt.title('Transition Dynamics: Investment')
                plt.xlabel('Time')
                plt.ylim([0.3, 0.4])
                plt.show()

            t3 = time.time()
            print(f'Plot time elapsed: {t3 - t2:.2f} seconds')

        t4 = time.time()
        print(f'\nTotal Run Time: {t4 - t0:.2f} seconds')

ncgm = Neoclassical_NN()
ncgm.value
ncgm.policy
ncgm.solve_model()