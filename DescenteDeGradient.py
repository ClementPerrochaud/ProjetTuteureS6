import time

def evalGrad(X:list, Y:list, model:'function', Coefs:list, loss0:float, dx:float) -> list:
    Grad = []
    for i in range(len(Coefs)):
        Coefs[i] += dx
        Grad.append((Loss(X, Y, model, Coefs) - loss0)/dx)
        Coefs[i] -= dx
    return Grad

def Loss(X:list, Y:list, model:'function', Coefs:list) -> float:
    '''La fonction model doit être de la forme  model(x:float, Coefs:list) -> float'''
    return sum([ (model(x,Coefs) - y)**2 for x,y in zip(X,Y)])/len(X)


# ALGORITHMES - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def basic(X:list, Y:list, model:'function', Coefs:list, 
          Loss:'function'=Loss, iterations:int=1000,
          gamma:float=1.E-5, dx:float=1.E-5) -> list:
    
    t0 = time.perf_counter()
    losses, times = [], []

    Coefs = Coefs.copy()
    l = len(Coefs)
    for n in range(1,iterations):
        loss0 = Loss(X,Y,model,Coefs)
        Grad  = evalGrad(X, Y, model, Coefs, loss0, dx)
        for i in range(l): Coefs[i] -= gamma*Grad[i]
        if n%(iterations//10)==0: print(f"{int(100*n/iterations)}%")

        losses.append(loss0)
        times.append(time.perf_counter()-t0)
    
    return Coefs, losses, times


def momentum(X:list, Y:list, model:'function', Coefs:list, 
             Loss:'function'=Loss, iterations:int=1000,
             gamma:float=1.E-4, beta:float=0.999, dx:float=1.E-5) -> list:
    
    t0 = time.perf_counter()
    losses, times = [], []

    Coefs = Coefs.copy()
    l = len(Coefs)
    V = [0]*l
    for n in range(1,iterations):
        loss0 = Loss(X,Y,model,Coefs)
        Grad  = evalGrad(X, Y, model, Coefs, loss0, dx)
        for i in range(l):
            V[i]      = beta*V[i] + (1-beta)*Grad[i]
            Coefs[i] -= gamma*V[i]
        if n%(iterations//10)==0: print(f"{int(100*n/iterations)}%")

        losses.append(loss0)
        times.append(time.perf_counter()-t0)
    
    return Coefs, losses, times


def AdaGrad(X:list, Y:list, model:'function', Coefs:list, 
            Loss:'function'=Loss, iterations:int=1000,
            gamma:float=1.E-1, eps:float=1.E-5, dx:float=1.E-5) -> list:
    
    t0 = time.perf_counter()
    losses, times = [], []

    Coefs = Coefs.copy()
    l = len(Coefs)
    G = [0]*l
    for n in range(1,iterations):
        loss0 = Loss(X,Y,model,Coefs)
        Grad  = evalGrad(X, Y, model, Coefs, loss0, dx)
        for i in range(l):
            G[i]  += Grad[i]**2
            Coefs[i] -= gamma/(G[i]**0.5+eps)*Grad[i]
        if n%(iterations//10)==0: print(f"{int(100*n/iterations)}%")

        losses.append(loss0)
        times.append(time.perf_counter()-t0)
    
    return Coefs, losses, times


def RMSprop(X:list, Y:list, model:'function', Coefs:list, 
            Loss:'function'=Loss, iterations:int=1000,
            gamma:float=1.E-3, beta:float=0.97, eps:float=1.E-5, dx:float=1.E-5) -> list:
    
    t0 = time.perf_counter()
    losses, times = [], []

    Coefs = Coefs.copy()
    l = len(Coefs)
    V = [0]*l
    for n in range(1,iterations):
        loss0 = Loss(X,Y,model,Coefs)
        Grad  = evalGrad(X, Y, model, Coefs, loss0, dx)
        for i in range(l):
            V[i]      = beta*V[i] + (1-beta)*Grad[i]**2
            Coefs[i] -= gamma/(V[i]**0.5+eps)*Grad[i]
        if n%(iterations//10)==0: print(f"{int(100*n/iterations)}%")

        losses.append(loss0)
        times.append(time.perf_counter()-t0)
    
    return Coefs, losses, times


def Adam(X:list, Y:list, model:'function', Coefs:list, 
         Loss:'function'=Loss, iterations:int=1000,
         gamma:float=1.E-2, beta:float=0.99, beta2:float=0.98,
         eps:float=1.E-5, dx:float=1.E-5) -> list:
    
    t0 = time.perf_counter()
    losses, times = [], []

    Coefs = Coefs.copy()
    l = len(Coefs)
    M = [0]*l
    V = [0]*l
    for n in range(1,iterations):
        loss0 = Loss(X,Y,model,Coefs)
        Grad  = evalGrad(X, Y, model, Coefs, loss0, dx)
        for i in range(l):
            M[i] = beta *M[i] + (1-beta) *Grad[i]
            V[i] = beta2*V[i] + (1-beta2)*Grad[i]**2
            m_hat = M[i]/(1-beta **n)
            v_hat = V[i]/(1-beta2**n)
            Coefs[i] -= gamma*m_hat/(v_hat**0.5+eps)
        if n%(iterations//10)==0: print(f"{int(100*n/iterations)}%")

        losses.append(loss0)
        times.append(time.perf_counter()-t0)
    
    return Coefs, losses, times


# TEST  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from math import log10
    import random

    def f(x, Coefs): # fonction polynome
        return sum([c*x**n for n,c in enumerate(Coefs)])
    
    degree = 20 # NOMBRE DE PARAMETRES         <-
    b,n = 1,15 # borne et nombre de points
    N = int(1E5) # nombre d'itterations        <-

    random.seed(0)
    CoefsRef = [random.gauss() for _ in range(degree)] # coefs a approcher
    Xref = [b*(2*i/n-1) for i in range(n)]
    Yref = [f(x, CoefsRef) for x in Xref]

    plot_type = "step" # "step" ou "time"      <-


    X = [[log10(1+i) for i in range(N-1)]]*5
    Coefs = [random.gauss() for _ in range(degree)]

    algo = [basic, momentum, AdaGrad, RMSprop, Adam]
    L, T = [], []
    print("Les calculs peuvent prendre un certain temps...")
    for i,optimiser in enumerate(algo):
        truc, losses, times = optimiser(Xref, Yref, f, Coefs, Loss, N)
        L.append([log10(l) for l in losses])
        T.append([log10(1+t) for t in times])
        print(f"100% : {i+1}/5 {optimiser.__name__}")

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(f"Évolution de recherche de la Loss pour un polynome de degré N={degree}\npour différents algorithmes de descente de gradient.")
    if   plot_type == "step": label = "log(n)"; Abscissa = X
    elif plot_type == "time": label = "log(t(s))"; Abscissa = T

    for i,(a,l,optimiser) in enumerate(zip(Abscissa,L,algo)):
        ax.plot(a, l, zorder=-i, label=optimiser.__name__)
    ax.set_xlabel(label)
    ax.grid(True)
    ax.set_ylabel("log(Loss(Θn))")
    ax.legend()

    plt.show()