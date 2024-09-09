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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ValueNetwork(nn.Module):
    def __init__(self, hidden_size=24):
        super(ValueNetwork, self).__init__()
        self.dense1 = nn.Linear(2, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dense3 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.dense1(x)
        x = self.Tanh(x)
        x = self.dense2(x)
        x = self.Tanh(x)
        x = self.dense3(x)
        x = self.Tanh(x)
        x = self.output(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, hidden_size=24):
        super(PolicyNetwork, self).__init__()
        self.dense1 = nn.Linear(2, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dense3 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.dense1(x)
        x = self.Tanh(x)
        x = self.dense2(x)
        x = self.Tanh(x)
        x = self.dense3(x)
        x = self.Tanh(x)
        x = self.output(x)
        return x

def pretrain(network, optimizer, data_loader, grid_k,grid_z, epochs=100, lr=0.01):
  criterion = nn.MSELoss()  # 平均二乗誤差を損失関数として使用
  optimizer = optim.Adam(network.parameters(), lr=lr)  # Adamオプティマイザを使用

  for epoch in range(epochs):
    total_loss = 0
    for data in data_loader:
      inputs = data.to(device)  # データをGPUに転送
      targets = data[:, 1].to(device)  # ターゲットもGPUに転送
      optimizer.zero_grad()  # 勾配を初期化
      outputs = network(inputs).squeeze()  # ネットワークの出力を計算
      loss = criterion(outputs, targets)  # 損失を計算
      loss.backward()  # 勾配を計算
      optimizer.step()  # オプティマイザを使ってパラメータを更新

  # 最終的な出力をプロットする
  print("Final output plot after pretraining:")

  # grid_kを入力に使って出力をプロット
  inputs = make_data(grid_z, grid_k).to(device)  # grid_kをテンソルに変換しGPUに送る
  outputs = network(inputs).detach().cpu().numpy()  # ネットワークの出力を計算し、CPUに戻す

  # プロット
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
    self.plott = plott      # select 1 to make plots
    self.simulate = simulate  # select 1 to run simulation
    self.full_euler_error = full_euler_error  # select 1 to calculate euler error on entire grid
    self.value = ValueNetwork().to(device)  # モデルをGPUに転送
    self.policy = PolicyNetwork().to(device)  # モデルをGPUに転送
    self.optimizer_value = optim.Adam(self.value.parameters(), lr=0.001)
    self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=0.001)

    self.setup_parameters()
    self.setup_grid()
    self.setup_markov()

    # Adding dataset and dataloader creation
    self.batch_size = 128  # You can choose an appropriate batch size
    self.dataset = CapitalDataset(self.grid_k, self.grid_z)  # Create a dataset from the grid_k data
    self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    self.params = self.sigma, self.beta, self.delta, self.alpha, self.grid_k, self.grid_z, \
              self.Nz, self.Nk, self.pi, self.maxit, self.tol, self.epochs, self.batch_size

  def setup_parameters(self):
    self.sigma = 2    # crra coefficient
    self.beta = 0.95  # discount factor
    self.rho = (1 - self.beta) / self.beta  # discount rate
    self.delta = 0.1  # depreciation rate
    self.alpha = 1 / 3  # cobb-douglas coeffient

    # TFP process parameters
    self.rho_z = 0.9           # autocorrelation coefficient
    self.sigma_z = 0.04        # std. dev. of shocks at annual frequency
    self.Nz = 7                # number of discrete income states
    self.z_bar = 0             # constant term in continuous productivity process

    # Steady state solution
    self.k_ss = (self.alpha / (1 / self.beta - (1 - self.delta))) ** (1 / (1 - self.alpha))  # capital
    self.c_ss = self.k_ss ** self.alpha - self.delta * self.k_ss  # consumption
    self.i_ss = self.delta * self.k_ss  # investment
    self.y_ss = self.k_ss ** self.alpha  # output

    # VFI solution
    self.tol = 1e-6  # tolerance for vfi
    self.maxit = 2000  # maximum number of iterations
    self.epochs = 1000  # number of epochs for pretraining
    self.Nk = 128*5
    self.dev = 0.9
    self.k_min = (1 - self.dev) * self.k_ss
    self.k_max = (1 + self.dev) * self.k_ss

    # Simulation
    self.seed = 123
    self.simT = 200    # number of simulation periods

  def setup_grid(self):
    # Capital grid
    self.grid_k = np.linspace(self.k_min, self.k_max, self.Nk)

  def setup_markov(self):
    self.mc = qe.markov.approximation.rouwenhorst(n=self.Nz, rho=self.rho_z, sigma=self.sigma_z, mu=self.z_bar)
    self.pi = self.mc.P
    self.grid_z = np.exp(self.mc.state_values)
    self.sim_z = np.exp(self.mc.simulate(self.simT, init=np.log(self.grid_z[int(self.Nz / 2)]), num_reps=None, random_state=self.seed))
    self.sim_z_idx = self.mc.simulate_indices(self.simT, init=int(self.Nz / 2), num_reps=None, random_state=self.seed)

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
    print(f"Time: {t1 - t0:.2f} seconds")

def utility(c, sigma):
  eps = torch.tensor(1e-8, dtype=torch.float64).to(device)

  if sigma == 1:
    return torch.log(torch.max(c, eps))
  else:
    return (torch.max(c, eps) ** (1 - sigma) - 1) / (1 - sigma)

def next(k_next, grid_z, value, batch, data, pi):
  next_value = torch.zeros(batch).to(device)
  for i in range(batch):
    k_next_i = k_next[i].repeat(7).to(device)
    next_data = torch.stack((grid_z, k_next_i), dim=1).to(device)
    z = data[i, 0].to(device)
    index = find_p(z, VFI)
    p = torch.tensor(pi[index, :], dtype=torch.float64).to(device)
    next_value[i] = torch.dot(value(next_data).squeeze(), p)
  return next_value

def find_p(p, VFI):
  z = torch.tensor(VFI.grid_z, dtype=torch.float64).to(device)
  index = (z == p).nonzero(as_tuple=True)[0].item()
  return index

def make_data_samez(grid_z, grid_k, index):
  grid_k = torch.tensor(grid_k, dtype=torch.float64).to(device)
  num_samples = len(grid_k)
  selected_z = grid_z[index]
  grid_z_repeated = torch.full((num_samples,), selected_z, dtype=torch.float64).to(device)
  data = torch.stack((grid_z_repeated, grid_k), dim=1)
  return data

def make_data(grid_z, grid_k):
  grid_k = torch.tensor(grid_k, dtype=torch.float64).to(device)
  num_samples = len(grid_k)
  grid_z = torch.tensor(grid_z[torch.randint(0, len(grid_z), (num_samples,))], dtype=torch.float64).to(device)
  data = torch.stack((grid_z, grid_k), dim=1)
  return data

def vfi_sto(params, value, policy, optimizer_value, optimizer_policy, dataloader):
  sigma, beta, delta, alpha, grid_k, grid_z, Nz, Nk, pi, maxit, tol, epochs, batch = params
  grid_z = torch.tensor(grid_z, dtype=torch.float64).to(device)
  grid_k = torch.tensor(grid_k, dtype=torch.float64).to(device)
  pi = torch.tensor(pi, dtype=torch.float64).to(device)

  for t in range(epochs):
    diff = 0
    count = 0
    for data in dataloader:
      data = data.to(device)  # データをGPUに転送
      count += 1
      pre_value = value(data)
      max_c = data[:, 0] * data[:, 1]**alpha + (1 - delta) * data[:, 1]
      c = torch.clamp(policy(data).squeeze(), min=torch.tensor(0, dtype=torch.float64).to(device), max=max_c)
      u = utility(c, sigma)
      u[c < 0] = -10e10
      k_next = max_c - c
      next_value = next(k_next, grid_z, value, batch, data, pi)
      bellman = u + beta * next_value.detach()

      loss = (value(data).squeeze() - bellman).pow(2).mean()
      optimizer_value.zero_grad()
      loss.backward(retain_graph=True)
      optimizer_value.step()

      next_value_pol = next(k_next, grid_z, value, batch, data, pi)
      bellman_policy = u + beta * next_value_pol
      loss_policy = -torch.mean(bellman_policy)
      optimizer_policy.zero_grad()
      loss_policy.backward()
      optimizer_policy.step()

      diff = max(diff, torch.abs(value(data) - pre_value).mean().item())

      if count == 5:
        print(f"Epoch {t} with diff {loss.item()}")

      if diff < tol:
        print(f"Converged at epoch {t+1} with diff {diff}")
        break
  data = make_data(grid_z, grid_k)
  consumption = policy(data).squeeze().detach().cpu().numpy()
  return consumption, t

VFI = Neoclassical_nn_stochastic()
VFI.value.double().to(device)
VFI.policy.double().to(device)
VFI.solve_model()

inputs = make_data_samez(VFI.grid_z, VFI.grid_k, 0).to(device)
plt.plot(VFI.grid_k, VFI.value(inputs).detach().cpu().numpy())
plt.title("value")
plt.xlabel('Input (grid_k)')
plt.ylabel('value')
plt.legend()
plt.show()

plt.plot(VFI.grid_k, VFI.policy(inputs).detach().cpu().numpy())
plt.plot(VFI.grid_k, VFI.grid_k, linestyle='--', label='Grid_k (Target)')
plt.title('policy')
plt.xlabel('Input (grid_k)')
plt.ylabel('consumption')
plt.legend()
plt.show()