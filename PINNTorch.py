import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
    
# RESEAU  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class FCN(nn.Module):
    def __init__(self, sizes, activation=nn.Sigmoid): 
        # activation fournie par Torch, eg: nn.Sigmoid, nn.Tanh, nn.ReLU... etc
        super().__init__()
        self.fcs = nn.Sequential(
                        nn.Linear(sizes[0], sizes[1]),
                        activation())
        self.fch = nn.Sequential(*[
                        nn.Sequential(
                            nn.Linear(sizes[i-1], sizes[i]),
                            activation()) 
                            for i in range(2,len(sizes)-1)])
        self.fce = nn.Linear(sizes[-2], sizes[-1])
    
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

    
torch.manual_seed(0)


# OSCILLATEUR HARMONIQUE AMORTI - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

m = 1 # kg
g = 7 # kg.s-1 (gamma)
k = 300 # N.m-1
x0 = 1 # m
v0 = 0 # m.s-1

def solution(t):
    D = g**2 - 4*m*k # kg2
    w = np.sqrt(np.abs(D))/(2*m) # rad.s-1
    T = 2*m/g # s-1
    if D > 0:
        A = (( 1/T+w)*x0 +v0)/(2*w)
        B = ((-1/T+w)*x0 -v0)/(2*w)
        return torch.exp(-t/T)*(A*torch.exp(w*t)+B*torch.exp(-w*t))
    elif D == 0:
        A = v0 + x0/T
        B = x0
        return (A*t+B)*torch.exp(-t/T)
    else:
        A=x0
        B=(v0+x0/T)/w
        return torch.exp(-t/T)*(A*torch.cos(w*t)+B*torch.sin(w*t))


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

print("initialisation...")
PINN = FCN((1,20,20,1))
optimiser = torch.optim.Adam(PINN.parameters())

T_ref   = torch.linspace(0,1,1 ).view(-1,1).requires_grad_(True)
T_coloc = torch.linspace(0,1,30).view(-1,1).requires_grad_(True)
T  = torch.linspace(0,1,300).view(-1,1)
Xs = solution(T)
#X0 = PINN(T)

n = 0
while True:
    
    X = PINN(T)
    fig, ax = plt.subplots(figsize=(19, 5))

    ax.plot(T.detach()[:,0], Xs.detach()[:,0], color='blue', linestyle='--', zorder=1)
   #ax.plot(T.detach()[:,0], X0.detach()[:,0], color='lightgrey',            zorder=0)

    ax.scatter(T_coloc.detach()[:,0], [0]*len(T_coloc), color='pink', zorder=1)
    ax.scatter(T_ref.detach()[:,0],  [x0],              color='blue', zorder=3)

    ax.plot(T.detach()[:,0], X.detach()[:,0], color='red', zorder=4)

    ax.set_xlabel("[t]")
    ax.set_ylabel("NN([t])")
    ax.grid(True)   
    plt.show()
   #plt.savefig(f"plot_{n}.png")

    for i in range(10_000):
        optimiser.zero_grad()

        X  = PINN(T_ref)
        dX = torch.autograd.grad(X, T_ref, torch.ones_like(X), create_graph=True)[0]
        lossExpX = (torch.squeeze(X)  - x0)**2 
        lossExpV = (torch.squeeze(dX) - v0)**2

        X = PINN(T_coloc)
        dX  = torch.autograd.grad(X,  T_coloc, torch.ones_like(X),  create_graph=True)[0]
        ddX = torch.autograd.grad(dX, T_coloc, torch.ones_like(dX), create_graph=True)[0]
        lossPhy = torch.mean((m*ddX + g*dX + k*X)**2)

        loss = lossPhy/1000 + lossExpX + lossExpV/10
        loss.backward()
        optimiser.step()
        print(f"{n} :\t{loss}"); n+=1
    
   #X0 = PINN(T)