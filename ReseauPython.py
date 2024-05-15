from math import exp, log, tanh # tanh exportee comme fonction d'activation usuelle
import random


# definition de quelques fonctions d'activation ( & tanh )
def sigmoid(x:float)    -> float: return 1/(1+exp(-x))
def ReLU(x:float)       -> float: return x if x>0 else 0
def Heaviside0(x:float) -> float: return 1 if x>0 else 0
def logLike(x:float)    -> float: return log(1+x) if x>0 else -log(1-x)


# RESEAU  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class NeuralNetwork:

    def __init__(self, sizes:tuple[int], activation_function:'function'=sigmoid, 
                 last_activation:bool=False, Coefs:list[float]=None, random_seed:int=0):
        
        self.sizes = sizes # forme du reseau de neurones
        self.aF = activation_function # fonction d'activation
        self.last_activation = last_activation # derniere couche affectee ?
        self.save_path = "Coefs.txt"

        self.len = 0    # nombre total de coefficients
        self.ind_w = {} # tableaux d'indices de w et b
        self.ind_b = {} # pointant le coefficient correspondant
        for i in range(1,len(sizes)):
            self.ind_w[i] = [[ self.len + sizes[i-1]*j + k for k in range(sizes[i-1])] for j in range(sizes[i])]
            self.ind_b[i] = [  self.len + sizes[i-1]*sizes[i] + j                      for j in range(sizes[i])]
            self.len += (sizes[i-1]+1)*sizes[i]

        # liste des coefficients
        random.seed(random_seed)
        if Coefs == None : self.Coefs = [random.gauss() for _ in range(self.len)]
        else:              self.Coefs = Coefs.copy()

    
    def evalGrad(self, loss0, loss_args):
        Grad = []
        for i in range(self.len):
            self.Coefs[i] += self.dx
            Grad.append((self.Loss(self,self.Coefs,*loss_args)-loss0)/self.dx)
            self.Coefs[i] -= self.dx
        return Grad

            
    # sauvegarder ou exporter des coefficients
    def import_Coefs(self, path:str=None):
        if path == None: path = self.save_path
        self.Coefs = [float(layer.strip("\n")) for layer in open(path).readlines()]
    def write_Coefs(self, path:str=None):
        if path == None: path = self.save_path
        file = open(path, "w")
        for x in self.Coefs: file.write(str(x)+"\n")


    def __call__(self, x:list[float], Coefs:list[float]=None) -> list[float]:
        if Coefs == None: Coefs = self.Coefs
        for i in range(1, len(self.sizes)): # propagation avant
            if self.last_activation or i != len(self.sizes)-1:
                x = [ self.aF( sum([ Coefs[self.ind_w[i][j][k]]*x[k] for k in range(self.sizes[i-1]) ]) + Coefs[self.ind_b[i][j]] ) for j in range(self.sizes[i])]
            else: x = [        sum([ Coefs[self.ind_w[i][j][k]]*x[k] for k in range(self.sizes[i-1]) ]) + Coefs[self.ind_b[i][j]]   for j in range(self.sizes[i])]
        return x
    # dans la definition precedente,  Coefs[ self.ind_w[i][j][k] ]  est w_ijk
    #                          idem,  Coefs[ self.ind_b[i][j]    ]  est b_ij


    # OPTIMISEURS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # NB: la Loss doit etre de la forme Loss(NN, Coefs, ...)

    ### init
    def basic_init(self, Loss:'function', gamma:float=1.E-5, dx:float=1.E-5):
        self.Loss = Loss
        self.parameters = [gamma]
        self.dx = dx
    def momentum_init(self, Loss:'function', gamma:float=1.E-4, beta:float=0.999, dx:float=1.E-5):
        self.Loss = Loss
        self.parameters = [gamma, beta]
        self.dx = dx
    def AdaGrad_init(self, Loss:'function', gamma:float=1.E-1, eps:float=1.E-5, dx:float=1.E-5):
        self.Loss = Loss
        self.parameters = [gamma, eps]
        self.dx = dx
    def RMSprop_init(self, Loss:'function', gamma:float=1.E-3, beta:float=0.97, eps:float=1.E-5, dx:float=1.E-5):
        self.Loss = Loss
        self.parameters = [gamma, beta, eps]
        self.dx = dx
    def Adam_init(self, Loss:'function', gamma:float=1.E-2, beta:float=0.99, beta2:float=0.98, eps:float=1.E-5, dx:float=1.E-5):
        self.Loss = Loss
        self.parameters = [gamma, beta, beta2, eps]
        self.dx = dx

    ### algorithmes

    def basic(self, iterations:int, *loss_args):

        gamma, = self.parameters
        loss0 = self.Loss(self,self.Coefs,*loss_args)
        best_loss = loss0
        best_Coefs = self.Coefs

        for n in range(iterations):
            Grad = self.evalGrad(loss0, loss_args)
            for i in range(self.len):
                self.Coefs[i] -= gamma*Grad[i]

            loss0 = self.Loss(self,self.Coefs,*loss_args)
            if loss0 < best_loss : best_loss = loss0; best_Coefs = self.Coefs
            print(f"{n} :\t{loss0}")

        self.Coefs = best_Coefs
        self.write_Coefs()


    def momentum(self, iterations:int, *loss_args):

        V = [0]*len(self.Coefs)
        gamma, beta = self.parameters
        loss0 = self.Loss(self,self.Coefs,*loss_args)
        best_loss = loss0
        best_Coefs = self.Coefs

        for n in range(iterations):
            Grad = self.evalGrad(loss0, loss_args)
            for i in range(self.len):
                V[i] = beta*V[i] + (1-beta)*Grad[i]
                self.Coefs[i] -= gamma*V[i]

            loss0 = self.Loss(self,self.Coefs,*loss_args)
            if loss0 < best_loss : best_loss = loss0; best_Coefs = self.Coefs
            print(f"{n} :\t{loss0}")

        self.Coefs = best_Coefs
        self.write_Coefs()


    def AdaGrad(self, iterations:int, *loss_args):

        G = [0]*len(self.Coefs)
        gamma, eps = self.parameters
        loss0 = self.Loss(self,self.Coefs,*loss_args)
        best_loss = loss0
        best_Coefs = self.Coefs

        for n in range(iterations):
            Grad = self.evalGrad(loss0, loss_args)
            for i in range(self.len):
                G[i] += Grad[i]**2
                self.Coefs[i] -= gamma/(G[i]**0.5 + eps)*Grad[i]

            loss0 = self.Loss(self,self.Coefs,*loss_args)
            if loss0 < best_loss : best_loss = loss0; best_Coefs = self.Coefs
            print(f"{n} :\t{loss0}")

        self.Coefs = best_Coefs
        self.write_Coefs()


    def RMSprop(self, iterations:int, *loss_args):

        V = [0]*len(self.Coefs)
        gamma, beta, eps = self.parameters
        loss0 = self.Loss(self,self.Coefs,*loss_args)
        best_loss = loss0
        best_Coefs = self.Coefs

        for n in range(iterations):    
            Grad = self.evalGrad(loss0, loss_args)
            for i in range(self.len):
                V[i] = beta*V[i] + (1-beta)*Grad[i]**2
                self.Coefs[i] -= gamma/(V[i]**0.5 + eps)*Grad[i]

            loss0 = self.Loss(self,self.Coefs,*loss_args)
            if loss0 < best_loss : best_loss = loss0; best_Coefs = self.Coefs
            print(f"{n} :\t{loss0}")

        self.Coefs = best_Coefs
        self.write_Coefs()


    def Adam(self, iterations:int, *loss_args):

        M = [0]*len(self.Coefs)
        V = [0]*len(self.Coefs)
        gamma, beta, beta2, eps = self.parameters
        loss0 = self.Loss(self,self.Coefs,*loss_args)
        best_loss = loss0
        best_Coefs = self.Coefs

        for n in range(1,iterations+1):
            Grad = self.evalGrad(loss0, loss_args)
            for i in range(self.len):
                M[i]  = beta *M[i] + (1-beta )*Grad[i]
                V[i]  = beta2*V[i] + (1-beta2)*Grad[i]**2
                m_hat = M[i]/(1-beta **n)
                v_hat = V[i]/(1-beta2**n)
                self.Coefs[i] -= gamma*m_hat/(v_hat**0.5 + eps)
            
            loss0 = self.Loss(self,self.Coefs,*loss_args)
            if loss0 < best_loss : best_loss = loss0; best_loss = loss0; best_Coefs = self.Coefs
            print(f"{n} :\t{loss0}")

        self.Coefs = best_Coefs
        self.write_Coefs()


# TEST  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    from math import cosh, sqrt
    
    # definition du reseau de neurones
    NN = NeuralNetwork((1,10,10,1), random_seed=4)

    # definition des points d'etude
    def functionRef(x):
        y = (cosh(x)-x**2+x)/5 - 10
        return y - y**2 / sqrt(y**2 + 5)
    nRef = 30
    Xref = [-10+20*i/nRef for i in range(nRef)]
    Yref = [functionRef(x) for x in Xref]

    # definition des abscisse et graphe initial du reseau
    X  = [ -10 + 20*i/500 for i in range(500)]
    Y0 = [NN([x])[0] for x in X]

    # definition de l'optimiseur (dont Loss) et optimisation
    def Loss(NN, Coefs, Xref, Yref): # la Loss doit etre de la forme Loss(NN, Coefs, ...) !
        return sum([ (NN([x],Coefs)[0] - y)**2 for x,y in zip(Xref,Yref) ])/len(Xref)
    print("calculs en cours...")
    NN.Adam_init(Loss)
    NN.Adam(2000, Xref, Yref) #       <-

    # graphe final
    Y = [NN([x])[0] for x in X]

    # plot
    fig, ax = plt.subplots()
    ax.plot(X, Y0, color='lightgrey')
    ax.scatter(Xref, Yref, color='green')
    ax.plot(X, Y, color='red')
    ax.set_xlabel("[x]")
    ax.set_ylabel("NN([x])")
    ax.grid(True)
    plt.show()