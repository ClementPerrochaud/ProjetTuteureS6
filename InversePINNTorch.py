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

m     = 1 # kg
g_ref = 7 # kg.s-1 (gamma)
k     = 300 # N.m-1
x0 = 1 # m
v0 = 0 # m.s-1

def solution(t):
    D = g_ref**2 - 4*m*k # kg2
    w = np.sqrt(np.abs(D))/(2*m) # rad.s-1
    T = 2*m/g_ref # s-1
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

g = torch.nn.Parameter(torch.tensor([0], dtype=torch.float32, requires_grad=True))
g_list = []

PINN = FCN((1,20,20,1), nn.Tanh)
optimiser = torch.optim.Adam(list(PINN.parameters())+[g])

T_ref   = torch.linspace(0,1,30 ).view(-1,1)
T_coloc = torch.linspace(0,1,30 ).view(-1,1).requires_grad_(True)
T       = torch.linspace(0,1,300).view(-1,1)
X_ref = solution(T_ref) + torch.randn_like(T_ref)/20
Xs = solution(T) 
#X0 = PINN(T)

n = 0
for _ in range(15):
    
    X = PINN(T)
    fig, ax = plt.subplots(figsize=(19, 5))

    ax.plot(T.detach()[:,0], Xs.detach()[:,0], color='blue', linestyle='--', zorder=1)
   #ax.plot(T.detach()[:,0], X0.detach()[:,0], color='lightgrey',            zorder=0)

    ax.scatter(T_coloc.detach()[:,0], [0]*len(T_coloc),    color='pink', zorder=1)
    ax.scatter(T_ref.detach()[:,0],   X_ref.detach()[:,0], color='blue', zorder=3)

    ax.plot(T.detach()[:,0], X.detach()[:,0], color='red', zorder=4)

    ax.set_xlabel("[t]")
    ax.set_ylabel("NN([t])")
    ax.grid(True)   
   #plt.show()
    plt.savefig(f"plot_{n}.png")

    
    for i in range(10_000):
        optimiser.zero_grad()

        X = PINN(T_ref)
        lossExp = torch.mean((X - X_ref)**2)

        X = PINN(T_coloc)
        dX  = torch.autograd.grad(X,  T_coloc, torch.ones_like(X),  create_graph=True)[0]
        ddX = torch.autograd.grad(dX, T_coloc, torch.ones_like(dX), create_graph=True)[0]
        lossPhy = torch.mean((m*ddX + g*dX + k*X)**2)

        loss = lossPhy/1000 + lossExp
        loss.backward()
        optimiser.step()
        g_list.append(g.item())
        print(f"{n} :\t{loss}\t g={g_list[-1]}"); n+=1
    
fig, ax = plt.subplots()
x = list(range(len(g_list)))
h = [g_ref for _ in x]
ax.plot(x, g_list, color="darkblue", zorder=2)
ax.plot(x, h, linestyle='--', color='goldenrod')
ax.set_xlabel("n")
ax.set_ylabel("gamma_n")
ax.grid(True)

plt.savefig("gamma.png")
plt.show()