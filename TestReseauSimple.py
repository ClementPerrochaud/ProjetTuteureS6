from math import exp, log, tanh # tanh exporee comme fonction d'activation usuelle
import random


# definition de quelques fonctions d'activation ( & tanh )
def sigmoid(x:float)    -> float: return 1/(1+exp(-x))
def ReLU(x:float)       -> float: return x if x>0 else 0
def Heaviside0(x:float) -> float: return 1 if x>0 else 0
def logLike(x:float)    -> float: return log(1+x) if x>0 else -log(1-x)


# RESEAU  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class NeuralNetwork:

    def __init__(self, sizes:tuple[int], activation_function:'function'=sigmoid, 
                 last_activation:bool=True, Coefs:list[float]=None, random_seed:int=0):

        self.sizes = sizes # forme du reseau de neurones
        self.aF = activation_function # fonction d'activation
        self.last_activation = last_activation # derniere couche affectee ?

        random.seed(random_seed)
        self.len = sum([ (sizes[i-1]+1)*sizes[i] for i in range(1,len(sizes)) ])
        if Coefs == None: # liste des coefficients
            Coefs = [ random.gauss() for _ in range(self.len) ]

        # definitions des matrices w et vecteurs b
        self.w = {}
        self.b = {}
        for i in range(1, len(sizes)):
            self.w[i] = [[Coefs.pop(0) for k in range(sizes[i-1])]  for j in range(sizes[i])]
            self.b[i] = [ Coefs.pop(0)                              for j in range(sizes[i])]
    

    def __call__(self, x:list[float]) -> list[float]:
        for i in range(1, len(self.sizes)): # propagation avant
            if self.last_activation or i != len(self.sizes)-1:
                x = [ self.aF( sum([ self.w[i][j][k]*x[k] for k in range(self.sizes[i-1]) ]) + self.b[i][j] ) for j in range(self.sizes[i])]
            else: x = [        sum([ self.w[i][j][k]*x[k] for k in range(self.sizes[i-1]) ]) + self.b[i][j]   for j in range(self.sizes[i])]
        return x


# TEST  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # definition des reseaux de neurone (1,...,1)
    sizes = [(1,2,2,1), (1,4,4,1), (1,5,5,5,1)]
    activ = [sigmoid, tanh, ReLU, logLike]
    Networks = [[NeuralNetwork(S,A,random_seed=1) for S in sizes] for A in activ]

    # abscisses & ordonnees[][]
    a,b,n = -8, 8, 1000 # bornes du plot & nombre de points
    X = [a+(b-a)*i/n for i in range(n)]
    Y = [[ [NN([x])[0] for x in X] for NN in NNl] for NNl in Networks]

    fig, ax = plt.subplots(2,2)
    fig.suptitle("Réseaux de neurones (1,...,1),\nici à coefficents aléatoirs (les mêmes pour chaque plot)\navec différentes fonctions d'activation.")
    titles = ["ϕ sigmoïde", "ϕ hyperbolique", "ϕ ReLU", "ϕ logarithmique"]
    for i,A in enumerate(activ):
        for j,S in enumerate(sizes):
            ax[i//2][i%2].plot(X, Y[i][j], linewidth=3, zorder=-j, label=str(S))
        ax[i//2][i%2].set_title(titles[i])
        ax[i//2][i%2].grid(True)
        ax[i//2][i%2].set_xlabel("[x]")
        ax[i//2][i%2].set_ylabel("NN([x])")
        ax[i//2][i%2].legend()

    plt.show()