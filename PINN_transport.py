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

# Net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Lin1 = nn.Linear(1, 20)
        self.Lin2 = nn.Linear(20, 20)
        self.Lin3 = nn.Linear(20, 20)
        self.Lin4 = nn.Linear(20, 20)
        self.out = nn.Linear(20, 1)
            
    def forward(self, x, k):
        out_flux = torch.sigmoid(self.Lin1(x))
        out_flux = torch.sigmoid(self.Lin2(out_flux))
        out_flux = torch.sigmoid(self.Lin3(out_flux))
        out_flux = torch.sigmoid(self.Lin4(out_flux))
        out_flux = self.out(out_flux)

        out_eigv = torch.sigmoid(self.Lin1(k))
        out_eigv = torch.sigmoid(self.Lin2(out_eigv))
        out_eigv = torch.sigmoid(self.Lin3(out_eigv))
        out_eigv = torch.sigmoid(self.Lin4(out_eigv))
        out_eigv = self.out(out_eigv)
        return out_flux, out_eigv


def Loss_PDE(x, k, D, S_a, vS_f, net):
    global NUM_COLLOCATION
    phi, k = net(x, k)
    phi_x = torch.autograd.grad(phi, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    phi_xx = torch.autograd.grad(phi_x, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    loss_pde = (S_a * phi) - (vS_f * phi) / k - (D * phi_xx)

    loss_f = nn.MSELoss()
    loss = loss_f(loss_pde, torch.zeros_like(loss_pde))
    return loss/NUM_COLLOCATION

def Loss_Regularization(x, k, vS_f, net):
    global NUM_COLLOCATION
    phi, k = net(x, k)
    if (phi.sum() < NUM_COLLOCATION) :
        loss_reg = (phi.sum() - NUM_COLLOCATION)
        return (loss_reg**2) / NUM_COLLOCATION
    else:
        return torch.sum(torch.zeros(1))

def Loss_Reflective(x, k, net):
    phi, k = net(x, k)
    phi_x = torch.autograd.grad(phi, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    loss_f = (phi_x.sum())**2
    return loss_f/2

def Loss_Fluxzero(x, k, net):
    phi, k = net(x, k)
    loss = (phi.sum())**2
    return loss/2

# 1D Geometry
LENGTH = 50.0
NUM_COLLOCATION = 200

# setting networks
net = Net()
net = net.to(device)
best_score = 99999999.0
best_model = net
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
iterations = 20000

# Initialization
x_test = np.linspace(0, LENGTH, NUM_COLLOCATION).reshape(NUM_COLLOCATION, 1)
x_test = Variable(torch.from_numpy(x_test).float(), requires_grad=True).to(device)
k = torch.tensor([1.00]).to(device)
k.requires_grad = True

for epoch in range(iterations):
    optimizer.zero_grad()

    x_boundary_l = Variable(torch.tensor([[0.0]]), requires_grad=True).to(device)
    x_boundary_r = Variable(torch.tensor([[LENGTH]]), requires_grad=True).to(device)
    loss_boundary = Loss_Reflective(x_boundary_l, k, net) + Loss_Fluxzero(x_boundary_r, k, net)

    x_collocation = np.random.uniform(0, LENGTH, size = (NUM_COLLOCATION,1))
    x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
    Ds = torch.where(x_collocation < 40, 6.8, 5.8).to(device)
    S_a = torch.where(x_collocation < 40, 0.30, 0.01).to(device)
    vS_f = torch.where(x_collocation < 40, 0.65, 0.00).to(device)

    loss_pde = Loss_PDE(x_collocation, k, Ds, S_a, vS_f, net)
    loss_regularization = Loss_Regularization(x_collocation, k, vS_f, net)

    # Combining the loss functions
    loss = loss_boundary + loss_pde + loss_regularization

    if (loss < best_score) :
        best_score = loss
        best_model = net

    loss.backward()
    optimizer.step()

    if (epoch % 100 == 0) :
        with torch.autograd.no_grad():
            i, k = net(x_test, k)
            print(f'Epochs: {epoch:<5} | Loss: {loss.data:<9.7f} | Eigenvalue: {k.data.cpu().numpy()}')
            

# Show
phi_result, k = best_model(x_test, k)
phi_result = phi_result.data.cpu().numpy()
print(f'Eigenvalue: {k.data.cpu().numpy()}')
# Plot
plt.plot(phi_result)
plt.show()