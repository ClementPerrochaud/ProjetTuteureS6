# Les PINNs ne diferent des reseaux de neurones que nous avons implementes
# dans le precedent programme que par la Loss utilisee.

# EXEMPLE : OSCILLATEUR HARMONIQUE AMORTI

from ReseauPython import *
import matplotlib.pyplot as plt
from math import cos, sin

# definition du reseau de neurones
PINN = NeuralNetwork((1,10,10,1)    , random_seed=2)

m  =  1 # kg
mu =  7 # kg.s-1 (frottements)
k = 300 # N.m-1
x0 = 1 # m
v0 = 0 # m.s-1

def solution(t):
    D = mu**2 - 4*m*k          # kg2
    w = (abs(D))**(1/2)/(2*m) # rad.s-1
    T  = 2*m/mu                # s-1
    if D > 0:
        A = (( 1/T+w)*x0 +v0)/(2*w)
        B = ((-1/T+w)*x0 -v0)/(2*w)
        return exp(-t/T)*(A*exp(w*t)+B*exp(-w*t))
    elif D == 0:
        A = v0 + x0/T
        B = x0
        return (A*t+B)*exp(-t/T)
    else:
        A=x0
        B=(v0+x0/T)/w
        return exp(-t/T)*(A*cos(w*t)+B*sin(w*t))

dt = 1.E-5 # s
n_coloc, t0_coloc, t1_coloc = 30, 0,   1
n_xref,  t0_xref,  t1_xref  =  1, 0, 0.3
n_vref,  t0_vref,  t1_vref  =  1, 0,   1
T_coloc = [t0_coloc + (t1_coloc-t0_coloc)*i/n_coloc for i in range(n_coloc)]
Tx_ref  = [t0_xref  + (t1_xref -t0_xref )*i/n_xref  for i in range(n_xref )]
Tv_ref  = [t0_vref  + (t1_vref -t0_vref )*i/n_vref  for i in range(n_vref )]
X_ref   = [ solution(t)                    for t in Tx_ref]
V_ref   = [(solution(t+dt)-solution(t))/dt for t in Tv_ref]

# definition des abscisse et graphe initial du reseau
T  = [i/500 for i in range(500)]
X0 = [PINN([t])[0] for t in T]
Xs = [solution(t)  for t in T]

# definition de la Loss experimentale & physique
def Loss(PINN, Coefs, T_coloc, Tx_ref, Tv_ref, X_ref, V_ref):
    X = {} # tous les appels necessaires
    for t in T_coloc:
        X[t]    = PINN([t],   Coefs)[0]
        X[t+dt] = PINN([t+dt],Coefs)[0]
        X[t-dt] = PINN([t-dt],Coefs)[0]
    for t in Tv_ref:
        if not t in X: 
            X[t]    = PINN([t],   Coefs)[0]
            X[t+dt] = PINN([t+dt],Coefs)[0]
    for t in Tx_ref:
        if not t in X: X[t] = PINN([t],Coefs)[0]

    lossPhy = sum([(   m*( X[t+dt] - 2*X[t] + X[t-dt] )/dt/dt 
                    + mu*( X[t+dt] -   X[t]           )/dt 
                    +  k*(             X[t]           )  )**2 for t in T_coloc])/max(1,len(T_coloc))
    lossExpV = sum([( (X[t+dt]-X[t])/dt - v )**2 for t,v in zip(Tv_ref,V_ref)])/max(1,len(Tv_ref) )
    lossExpX = sum([(          X[t]     - x )**2 for t,x in zip(Tx_ref,X_ref)])/max(1,len(Tx_ref) )

    return lossPhy/1000 + lossExpX + lossExpV/10


while True:
    # initialisation de l'optimiseur et optimisation
    PINN.Adam_init(Loss)
    PINN.Adam(1000, T_coloc, Tx_ref, Tv_ref, X_ref, V_ref)

    # graphe final
    X = [PINN([t])[0] for t in T]

    # plot
    fig, ax = plt.subplots(figsize=(19, 5))

    ax.plot(T, Xs, color='blue', linestyle='--', zorder=1)
    ax.plot(T, X0, color='lightgrey',            zorder=0)

    ax.scatter(T_coloc, [0]*len(T_coloc), color='pink', zorder=1)
    ax.scatter(Tv_ref, V_ref, marker='s', color='cyan', zorder=2)
    ax.scatter(Tx_ref, X_ref,             color='blue', zorder=3)

    ax.plot(T, X, color='red', zorder=4)

    ax.set_xlabel("[t]")
    ax.set_ylabel("NN([t])")
    ax.grid(True)
    plt.show()

    X0 = [PINN([t])[0] for t in T]