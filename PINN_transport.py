import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable

# Based on the repository of: https://github.com/nanditadoloi/PINN/blob/main/solve_PDE_NN.ipynb
# Based on the paper of: https://doi.org/10.1080/00295639.2022.2123211
# Physics-Informed Neural Network Method and Application to Nuclear Reactor Calculations: A Pilot Study

# Solving 1D, 1G problem from HYU depeartment of nuclear engineering, reactor theory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

def weight_uniform(submodule):
    if isinstance(submodule, torch.nn.Linear):
        torch.nn.init.uniform_(submodule.weight, 0)
        submodule.bias.data.fill_(0)
# Net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(1, 8), nn.Sigmoid(),
            nn.Linear(8, 8), nn.Sigmoid(),
            nn.Linear(8, 8), nn.Sigmoid(),
            nn.Linear(8, 1)
        )
        self.main.apply(weight_uniform)
            
    def forward(self, x):
        output = self.main(x)
        return output

def r(x, NUM_COLLOCATION, net):
    global INIT_FLUX
    phi = net(x)
    #print(torch.sum(phi))
    loss_reg = (((torch.sum(phi)) - (INIT_FLUX*NUM_COLLOCATION))/NUM_COLLOCATION)**2
    return loss_reg

def f(x, D, S_a, vS_f, NUM_COLLOCATION, net):
    phi = net(x)
    phi_x = torch.autograd.grad(phi, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    phi_xx = torch.autograd.grad(phi_x, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    #print(torch.cat((phi, phi_x, phi_xx), dim=1))
    loss_pde = (S_a * phi) - (vS_f * phi) - (D * phi_xx)
    loss_f = nn.MSELoss()
    loss = loss_f(loss_pde, torch.zeros_like(loss_pde))
    return loss/NUM_COLLOCATION

def Reflective_boundary(x, net):
    phi = net(x)
    phi_x = torch.autograd.grad(phi, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    loss_f = nn.MSELoss()
    loss = loss_f(phi_x, torch.zeros_like(phi_x))
    return loss/2

def Fluxzero_boundary(x, net):
    phi = net(x)
    loss_bc = phi
    loss_f = nn.MSELoss()
    loss = loss_f(loss_bc, torch.zeros_like(loss_bc))
    return loss/2

def Vacuum_boundary(x, net):
    phi = net(x)
    loss_bc = phi
    loss_f = nn.MSELoss()
    loss = loss_f(loss_bc, torch.zeros_like(loss_bc))
    return loss/2

# 1D Geometry
LENGTH = 50.0
NUM_COLLOCATION = 20
INIT_FLUX = 0.5

# setting networks
net = Net()
net = net.to(device)
best_score = 99999999.0
best_model = net
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas = (0.9,0.99), eps = 10**-15)
iterations = 5000
eigval = 1.00

x_test = np.linspace(0, LENGTH, NUM_COLLOCATION).reshape(NUM_COLLOCATION, 1)
x_test = Variable(torch.from_numpy(x_test).float(), requires_grad=True).to(device)
F_test = np.where(x_test < 40.0, 0.65, 0.00)
phi_prior = np.linspace(INIT_FLUX, INIT_FLUX, NUM_COLLOCATION).reshape(NUM_COLLOCATION, 1)
print(phi_prior.sum())
for epoch in range(iterations):
    optimizer.zero_grad()

    x_bc1 = np.array([[0.0]])
    x_bc1 = Variable(torch.from_numpy(x_bc1).float(), requires_grad=True).to(device)
    mse_b1 = Reflective_boundary(x_bc1, net)

    x_bc2 = np.array([[LENGTH]])
    x_bc2 = Variable(torch.from_numpy(x_bc2).float(), requires_grad=True).to(device)
    mse_b2 = Fluxzero_boundary(x_bc2, net)

    x_collocation = np.random.uniform(0.02, LENGTH-0.02, (NUM_COLLOCATION,1))
    x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)

    Ds = torch.where(x_collocation < 40, 6.8, 5.8).to(device)
    S_a = torch.where(x_collocation < 40, 0.30, 0.01).to(device)
    vS_f = torch.where(x_collocation < 40, 0.65, 0.00).to(device)

    mse_f = f(x_collocation, Ds, S_a, vS_f/eigval, NUM_COLLOCATION, net)
    mse_r = r(x_collocation, NUM_COLLOCATION, net)

    # Combining the loss functions
    loss = mse_b1 + mse_b2 + mse_f + mse_r

    if (loss < best_score) :
        best_score = loss
        best_model = net

    if (epoch % 10 == 5) :
        phi_after = net(x_test).data.cpu().numpy()
        eigval = eigval * ((F_test * phi_after).sum() / (F_test * phi_prior).sum())
        phi_prior = phi_after

    loss.backward()
    optimizer.step()
    if (epoch % 100 == 0) :
        with torch.autograd.no_grad():
            print("Epoch:", epoch, " | Boundary loss:", mse_b1.data, mse_b2.data, " | Traning loss:", loss.data, " | Eigenvalue:", eigval)

# Show
print("Loss: ", best_score)
phi_result = best_model(x_test)
phi_result = phi_result.data.cpu().numpy()
x_test = np.linspace(0, LENGTH, NUM_COLLOCATION).reshape(NUM_COLLOCATION, 1)

plt.plot(x_test, phi_result)
plt.show()

# Save Model
torch.save(net.state_dict(), "model_uxt.pt")