import time
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import seaborn as sns
import quantecon as qe
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
sns.set(style='whitegrid')

class ValueNetwork(nn.Module):
  def __init__(self, hidden_size=12):
    super(ValueNetwork, self).__init__()
    self.dense1 = nn.Linear(2, hidden_size)
    self.dense2 = nn.Linear(hidden_size, hidden_size)
    self.dense3 = nn.Linear(hidden_size, 1)
    self.relu = nn.ReLU()
    self.Tanh = nn.Tanh()

  def forward(self, x):
    x = self.dense1(x)
    x = self.Tanh(x)
    x = self.dense2(x)
    x = self.Tanh(x)
    x = self.dense3(x)
    return x

class PolicyNetwork(nn.Module):
  def __init__(self, hidden_size=12):
    super(PolicyNetwork, self).__init__()
    self.dense1 = nn.Linear(2, hidden_size)
    self.dense2 = nn.Linear(hidden_size, hidden_size)
    self.dense3 = nn.Linear(hidden_size, 1)
    self.relu = nn.ReLU()
    self.Tanh = nn.Tanh()

  def forward(self, x):
    x = self.dense1(x)
    x = self.relu(x)
    x = self.dense2(x)
    x = self.relu(x)
    x = self.dense3(x)
    return x

def pretrain(network, optimizer, data_loader, grid_k,grid_z, epochs=100):
  criterion = nn.MSELoss()# 平均二乗誤差を損失関数として使用

  for epoch in range(epochs):
    total_loss = 0
    for data in data_loader:
      inputs = data  #dataloaderからのデータが入力
      targets = data[:, 1]  #プレトレーニングではgrid_k自体をターゲットとする
      optimizer.zero_grad()  #勾配を初期化
      outputs = network(inputs).squeeze()  #ネットワークの出力を計算
      loss = criterion(outputs, targets) #損失を計算
      loss.backward()  #勾配を計算
      optimizer.step()  #オプティマイザを使ってパラメータを更新


  #最終的な出力をプロットする
  print("Final output plot after pretraining:")

  #grid_kを入力に使って出力をプロット
  inputs = make_data(grid_z, grid_k) # grid_kをテンソルに変換
  outputs = network(inputs).detach().numpy()  #ネットワークの出力を計算

  #プロット
  plt.plot(grid_k, outputs, label='Pretrained Output')
  plt.plot(grid_k, grid_k, linestyle='--', label='Grid_k (Target)')
  plt.title('Network Output After Pretraining')
  plt.xlabel('Input (grid_k)')
  plt.ylabel('Output')
  plt.legend()
  plt.show()

class CapitalDataset(Dataset):
  def __init__(self, grid_k, grid_z):
    super(CapitalDataset, self).__init__()
    self.grid_k = torch.tensor(grid_k, dtype=torch.float64)
    self.num_samples = len(grid_k)
    self.grid_z = torch.tensor(grid_z[torch.randint(0, len(grid_z), (self.num_samples,))], dtype=torch.float64)
    self.data = torch.stack((self.grid_z,self.grid_k), dim=1)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]

class Neoclassical_nn_stochastic:
  def __init__(self, plott=1, simulate=1, full_euler_error=0):

    self.plott = plott      #select 1 to make plots
    self.simulate = simulate  #select 1 to run simulation
    self.full_euler_error = full_euler_error  #select 1 to calculate euler error on entire grid
    self.value = ValueNetwork()
    self.policy = PolicyNetwork()
    self.optimizer_value = optim.Adam(self.value.parameters(), lr=0.001)
    self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=0.001)

    self.setup_parameters()
    self.setup_grid()
    self.setup_markov()

    # Adding dataset and dataloader creation
    self.batch_size = 20  # You can choose an appropriate batch size
    self.dataset = CapitalDataset(self.grid_k, self.grid_z)  # Create a dataset from the grid_k data
    self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


    self.params = self.sigma, self.beta, self.delta, self.alpha, self.grid_k, self.grid_z, \
              self.Nz, self.Nk, self.pi, self.maxit, self.tol, self.epochs, self.batch_size

    if self.plott != 1 and self.plott != 0:
                raise Exception("Plot option incorrectly entered: Choose either 1 or 0.")

    if self.simulate != 1 and self.simulate != 0:
        raise Exception("Simulate option incorrectly entered: Choose either 1 or 0.")

    if self.full_euler_error != 1 and self.full_euler_error != 0:
        raise Exception("Euler error full grid evaluation option incorrectly entered: Choose either 1 or 0.")

  def setup_parameters(self):

    # a. model parameters
    self.sigma=2    #crra coefficient
    self.beta = 0.95  # discount factor
    self.rho = (1-self.beta)/self.beta #discount rate
    self.delta = 0.1  # depreciation rate
    self.alpha = 1/3  # cobb-douglas coeffient

    # b. tfp process parameters
    self.rho_z = 0.9           #autocorrelation coefficient
    self.sigma_z = 0.04        #std. dev. of shocks at annual frequency
    self.Nz = 7                #number of discrete income states
    self.z_bar = 0             #constant term in continuous productivity process (not the mean of the process)

    # c. steady state solution
    self.k_ss = (self.alpha/(1/self.beta - (1-self.delta)))**(1/(1 - self.alpha)) #capital
    self.c_ss = self.k_ss**(self.alpha) - self.delta*self.k_ss #consumption
    self.i_ss = self.delta*self.k_ss    #investment
    self.y_ss = self.k_ss**(self.alpha) #output

    # d. vfi solution
    self.tol = 1e-6  # tolerance for vfi
    self.maxit = 2000  # maximum number of iterations
    self.epochs = 300  # number of epochs for pretraining
    # capital grid
    self.Nk = 500
    self.dev = 0.9
    self.k_min = (1-self.dev)*self.k_ss
    self.k_max = (1+self.dev)*self.k_ss

    # e. simulation
    self.seed = 123
    self.cheb_order = 10    #chebyshev polynomial order
    self.simT = 200    #number of simulation periods

    # f. euler equation error analysis
    if self.full_euler_error:
        self.Nk_fine=2500

  def setup_grid(self):

    # a. capital grid
    self.grid_k = np.linspace(self.k_min, self.k_max, self.Nk)

    # b. finer grids for euler error analysis
    if self.full_euler_error:
        self.grid_k_fine = np.linspace(self.k_min, self.k_max, self.Nk_fine)

  def setup_markov(self):

    # a. discretely approximate the continuous income process
    self.mc = qe.markov.approximation.rouwenhorst(n=self.Nz, rho=self.rho_z, sigma=self.sigma_z, mu=self.z_bar)
    #self.mc = qe.markov.approximation.tauchen(n=self.Nz, rho=self.rho_z, sigma=self.sigma_z, mu=self.z_bar, n_std=3)

    # b. transition matrix and states
    self.pi = self.mc.P
    self.grid_z = np.exp(self.mc.state_values)

    # c. simulation

    #generate markov simulation for tfp
    self.sim_z =  self.mc.simulate(self.simT, init=np.log(self.grid_z[int(self.Nz/2)]), num_reps=None, random_state=self.seed)
    self.sim_z = np.exp(self.sim_z)

    # indices of grid_z over the simulation
    self.sim_z_idx = self.mc.simulate_indices(self.simT, init=int(self.Nz/2), num_reps=None, random_state=self.seed)

  def pretrain_models(self, pretrain_epochs=10):
    print("Pretraining Value Network...")
    pretrain(self.value, self.optimizer_value, self.dataloader, self.grid_k, self.grid_z, pretrain_epochs)
    print("Pretraining Policy Network...")
    pretrain(self.policy, self.optimizer_policy, self.dataloader, self.grid_k, self.grid_z, pretrain_epochs)

  def solve_model(self):
    t0 = time.time()
    self.pretrain_models()
    self.pol_cons, t = vfi_sto(self.params, self.value, self.policy, self.optimizer_value, self.optimizer_policy, self.dataloader)
    t1 = time.time()
    print(f"Time: {t1-t0:.2f} seconds")
    

def utility(c, sigma):
  eps = torch.tensor(1e-8, dtype=torch.float64)

  if sigma == 1:
      return torch.log(torch.max(c, eps))
  else:
      return (torch.max(c, eps)**(1-sigma) - 1) / (1 - sigma)

def next(k_next, grid_z, value,batch, data, pi):
  next_value = torch.zeros(batch)
  for i in range(batch):
    k_next_i = k_next[i].repeat(7)
    next_data = torch.stack((grid_z, k_next_i), dim=1)
    z = data[i, 0]
    index = find_p(z, VFI)
    p = torch.tensor(pi[index, :], dtype=torch.float64)
    next_value[i] = torch.dot(value(next_data).squeeze(), p)
  return next_value

def find_p(p, VFI):
  z = torch.tensor(VFI.grid_z, dtype=torch.float64) # zは7要素の配列（ベクトル）
  # pとzの各要素を比較して一致するインデックスを取得
  index = (z == p).nonzero(as_tuple=True)[0].item()  # 一致するインデックスを取得
  return index


def make_data_samez(grid_z, grid_k, index):
  # grid_kのテンソル化
  grid_k = torch.tensor(grid_k, dtype=torch.float64)
  num_samples = len(grid_k)

  # 指定されたindexのgrid_zの要素を取り出しそれを繰り返し
  selected_z = grid_z[index]  # 指定されたindexの要素
  grid_z_repeated = torch.full((num_samples,), selected_z, dtype=torch.float64)  # 繰り返し

  # grid_z_repeatedとgrid_kをスタック
  data = torch.stack((grid_z_repeated, grid_k), dim=1)

  return data

def make_data(grid_z, grid_k):
  grid_k = torch.tensor(grid_k, dtype=torch.float64)
  num_samples = len(grid_k)
  grid_z = torch.tensor(grid_z[torch.randint(0, len(grid_z), (num_samples,))], dtype=torch.float64)
  data = torch.stack((grid_z,grid_k), dim=1)

  return data

def vfi_sto(params, value, policy, optimizer_value, optimizer_policy, dataloader):
  sigma, beta, delta, alpha, grid_k, grid_z, Nz, Nk, pi, maxit, tol, epochs, batch = params
  grid_z = torch.tensor(grid_z, dtype=torch.float64)
  grid_k = torch.tensor(grid_k, dtype=torch.float64)
  pi = torch.tensor(pi, dtype=torch.float64)

  for t in range(epochs):
    diff = 0
    count = 0
    for data in dataloader:
      count += 1
      pre_value = value(data)
      max_c = data[:, 0] * data[:, 1]**alpha + (1-delta) * data[:, 1]# データの型注意
      c = torch.clamp(policy(data).squeeze(), min=torch.tensor(0, dtype=torch.float64), max=max_c)
      u = utility(c, sigma)
      u[c < 0] = -10e10
      k_next = max_c - c
      next_value = next(k_next, grid_z, value, batch, data, pi)
      bellman = u + beta * next_value.detach()

      loss = (value(data).squeeze() - bellman).pow(2).mean()
      optimizer_value.zero_grad()
      loss.backward(retain_graph = True)
      optimizer_value.step()

      next_value_pol = next(k_next, grid_z, value, batch, data, pi)
      bellman_policy = u + beta * next_value_pol
      loss_policy = -torch.mean(bellman_policy)
      optimizer_policy.zero_grad()
      loss_policy.backward()
      optimizer_policy.step()

      diff = max(diff, torch.abs(value(data) - pre_value).mean().item())

      if count == 10:
        print(f"Epoch {t} with diff {diff}")

      if diff < tol:
        print(f"Converged at epoch {t+1} with diff {diff}")
        break
  data = make_data(grid_z, grid_k)
  consumption = policy(data).squeeze().detach().numpy()
  return consumption, t

VFI = Neoclassical_nn_stochastic()
VFI.value.double()
VFI.policy.double()
VFI.solve_model()

inputs = make_data_samez(VFI.grid_z, VFI.grid_k,0)
plt.plot(VFI.grid_k, VFI.value(inputs).detach().numpy())
plt.title("value")
plt.xlabel('Input (grid_k)')
plt.ylabel('value')
plt.legend()
plt.show()

plt.plot(VFI.grid_k, VFI.policy(inputs).detach().numpy())
plt.plot(VFI.grid_k, VFI.grid_k, linestyle='--', label='Grid_k (Target)')
plt.title('policy')
plt.xlabel('Input (grid_k)')
plt.ylabel('consumption')
plt.legend()
plt.show()
