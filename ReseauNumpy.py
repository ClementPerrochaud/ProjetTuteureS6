import numpy as np # le changement de paradigme : la vectorisation
from math import tanh # tanh exportee comme fonction d'activation usuelle
import functools

# permet de faire fonctionner la vectorisation dans une methode de classe (__call__)
# Kurt Peek sur Stack Overflow
class vectorize(np.vectorize):
    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)

# definition de quelques fonctions d'activation ( & tanh )
def sigmoid(x:float)    -> float: return 1/(1+np.exp(-x))
def ReLU(x:float)       -> float: return np.maximum(0,x)
def Heaviside0(x:float) -> float: return np.maximum(0,np.sign(x))
def logLike(x:float)    -> float: return np.sign(x)*np.log(1+np.abs(x))


# RESEAU  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class NeuralNetwork:

    def __init__(self, sizes:tuple[int], activation_function:'function'=sigmoid, 
                 last_activation:bool=False, Coefs:list[float]=None, random_seed:int=0):
        
        self.sizes = sizes # forme du reseau de neurones
        self.aF = activation_function # fonction d'activation compatible numpy
        self.last_activation = last_activation # derniere couche affectee ?
        self.save_path = "Coefs.txt"

        self.len = 0
        self.ind_w = {} # "listes" d'indices de w et b
        self.ind_b = {} # pointe l'indice du premier coefficient de w_i et b_i
        for i in range(1,len(sizes)):
            self.ind_w[i] = self.len
            self.ind_b[i] = self.len + sizes[i-1]*sizes[i]
            self.len += (sizes[i-1]+1)*sizes[i]
        self.ind_w[len(sizes)] = self.len

        # liste des coefficients
        np.random.seed(random_seed)
        if Coefs == None : self.Coefs = np.random.normal(size=self.len)
        else:              self.Coefs = np.array(Coefs).copy()

            
    # sauvegarder ou exporter des coefficients
    def import_Coefs(self, path:str=None):
        if path == None: path = self.save_path
        self.Coefs = [float(layer.strip("\n")) for layer in open(path).readlines()]
    def write_Coefs(self, path:str=None):
        if path == None: path = self.save_path
        file = open(path, "w")
        for x in self.Coefs: file.write(str(x)+"\n")
        
    @vectorize(excluded=[0,2], signature='(1)->(1)')
    def __call__(self, x:'np.ndarray', Coefs:'np.ndarray'=None) -> 'np.ndarray':
        if Coefs is None: Coefs = self.Coefs
        x = np.vstack(x)
        for i in range(1, len(self.sizes)): # propagation avant
            w = Coefs[self.ind_w[i]:self.ind_b[i]  ].reshape((self.sizes[i],self.sizes[i-1]))
            b = Coefs[self.ind_b[i]:self.ind_w[i+1]].reshape((self.sizes[i],1              ))
            x = np.matmul(w,x)+b
            if self.last_activation or i != len(self.sizes)-1: x = self.aF(x)
        return x


    # OPTIMISEURS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # NB: la Loss doit etre de la forme Loss(NN, Coefs, ...)

    ### init

    def basic_init(self, Loss:'function', gamma:float=1.E-5, dx:float=1.E-5):
        self.Loss  = Loss
        self.gamma = gamma
        self.dx    = dx

    def momentum_init(self, Loss:'function', gamma:float=1.E-4, beta:float=0.999, dx:float=1.E-5):
        self.Loss  = Loss
        self.gamma = gamma
        self.beta  = beta
        self.dx    = dx

    def AdaGrad_init(self, Loss:'function', gamma:float=1.E-1, eps:float=1.E-5, dx:float=1.E-5):
        self.Loss  = Loss
        self.gamma = gamma
        self.eps   = eps
        self.dx    = dx

    def RMSprop_init(self, Loss:'function', gamma:float=1.E-3, beta:float=0.97, eps:float=1.E-5, dx:float=1.E-5):
        self.Loss  = Loss
        self.gamma = gamma
        self.beta  = beta
        self.eps   = eps
        self.dx    = dx

    def Adam_init(self, Loss:'function', gamma:float=1.E-2, beta1:float=0.99, beta2:float=0.98, eps:float=1.E-5, dx:float=1.E-5):
        self.Loss  = Loss
        self.gamma = gamma
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.dx    = dx


    ### usefull GradEval function
    def GradEval(self, loss0:float, loss_args) -> 'np.ndarray':
        L = []
        for i in range(self.len):
            C = self.Coefs.copy()
            C[i] += self.dx
            L.append(( self.Loss(self,C,*loss_args) - loss0)/self.dx )
        return np.array(L)
    
    #@vectorize(excluded=[0,2,3], signature='()->(1)')
    #def GradEval(self, i:int, loss0:float, *loss_args) -> float:
    #    C = self.Coefs.copy()
    #    C[i] += self.dx
    #    return ( self.Loss(self,C,*loss_args) - loss0)/self.dx
    
    
    ### algorithmes
    
    def basic(self, iterations:int, *loss_args):

        loss0 = self.Loss(self, self.Coefs, *loss_args)
        best_loss = loss0
        best_Coefs = self.Coefs

        for n in range(iterations):
            Grad        = self.GradEval(loss0,loss_args)
            self.Coefs -= self.gamma*Grad

            loss0 = self.Loss(self, self.Coefs, *loss_args)
            if loss0 < best_loss : best_loss = loss0; best_Coefs = self.Coefs
            print(f"{n} :\t{loss0}")

        self.Coefs = best_Coefs
        self.write_Coefs()


    def momentum(self, iterations:int, *loss_args):

        V = np.zeros(self.len)
        loss0 = self.Loss(self,self.Coefs,*loss_args)
        best_loss = loss0
        best_Coefs = self.Coefs


        for n in range(iterations):
            Grad        = self.GradEval(loss0,*loss_args)
            V           = self.beta*V + (1-self.beta)*Grad
            self.Coefs -= self.gamma*V

            loss0 = self.Loss(self,self.Coefs,*loss_args)
            if loss0 < best_loss : best_loss = loss0; best_Coefs = self.Coefs
            print(f"{n} :\t{loss0}")

        self.Coefs = best_Coefs
        self.write_Coefs()


    def AdaGrad(self, iterations:int, *loss_args):

        G = np.zeros(self.len)
        loss0 = self.Loss(self,self.Coefs,*loss_args)
        best_loss = loss0
        best_Coefs = self.Coefs

        for n in range(iterations):
            Grad        = self.GradEval(loss0,*loss_args)
            G          += Grad**2
            self.Coefs -= (self.gamma/(np.sqrt(G)+self.eps))*Grad

            loss0 = self.Loss(self,self.Coefs,*loss_args)
            if loss0 < best_loss : best_loss = loss0; best_Coefs = self.Coefs
            print(f"{n} :\t{loss0}")

        self.Coefs = best_Coefs
        self.write_Coefs()


    def RMSprop(self, iterations:int, *loss_args):

        V = np.zeros(self.len)
        loss0 = self.Loss(self,self.Coefs,*loss_args)
        best_loss = loss0
        best_Coefs = self.Coefs

        for n in range(iterations):    
            Grad        = self.GradEval(loss0,*loss_args)
            V           = self.beta*V + (1-self.beta)*Grad**2
            self.Coefs -= (self.gamma/(np.sqrt(V)+self.eps))*Grad

            loss0 = self.Loss(self,self.Coefs,*loss_args)
            if loss0 < best_loss : best_loss = loss0; best_Coefs = self.Coefs
            print(f"{n} :\t{loss0}")

        self.Coefs = best_Coefs
        self.write_Coefs()

    def Adam(self, iterations:int, *loss_args):

        M = np.zeros(self.len)
        V = np.zeros(self.len)
        loss0 = self.Loss(self,self.Coefs,*loss_args)
        best_loss  = loss0
        best_Coefs = self.Coefs

        for n in range(1,iterations+1):
            Grad        = self.GradEval(loss0,*loss_args)
            M           = self.beta1*M + (1-self.beta1)*Grad
            V           = self.beta2*V + (1-self.beta2)*Grad**2
            self.Coefs -= self.gamma * M/(1-self.beta1**n) / (np.sqrt( V/(1-self.beta2**n) )+self.eps)
            
            loss0 = self.Loss(self,self.Coefs,*loss_args)
            if loss0 < best_loss : best_loss = loss0; best_loss = loss0; best_Coefs = self.Coefs
            print(f"{n} :\t{loss0}")

        self.Coefs = best_Coefs
        self.write_Coefs()


# TEST  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    
    # definition du reseau de neurones
    NN = NeuralNetwork((1,10,10,1), activation_function=logLike, random_seed=4)

    # definition des points d'etude
    def functionRef(x): return (x**2 - 3*x)/(2*x**2 - x + 1)
    nRef = 15
    Xref = np.vstack(np.linspace(-3,3,nRef))
    Yref = functionRef(Xref)

    # definition des abscisse et graphe initial du reseau
    X  = np.vstack(np.linspace(-5,10,500))
    Y0 = NN(X)

    # definition de l'optimiseur (dont Loss) et optimisation
    def Loss(NN, Coefs, Xref, Yref): # la Loss doit etre de la forme Loss(NN, Coefs, ...) !
        return np.mean((NN(Xref,Coefs)-Yref)**2)
    print("calculs en cours...")
    NN.basic_init(Loss)
    t=time.perf_counter()
    NN.basic(100, Xref, Yref) #       <-
    t=time.perf_counter()-t
    print(f"{round(t,3)} secondes")

    # graphe final
    Y = NN(X)

    # plot
    fig, ax = plt.subplots()
    ax.plot(X, Y0, color='lightgrey')
    ax.scatter(Xref, Yref, color='green')
    ax.plot(X, Y, color='red')
    ax.set_xlabel("[x]")
    ax.set_ylabel("NN([x])")
    ax.grid(True)
    plt.show()