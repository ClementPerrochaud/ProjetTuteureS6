#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <random>
using namespace std;

// definition de quelques fonctions d'activation ( & tanh )
void aF_sigmoid(double& x)    { x = 1 / (1 + exp(-x)); }
void aF_tanh(double& x)       { x = tanh(x); }
void aF_ReLU(double& x)       { x = x > 0 ? x : 0; }
void aF_Heaviside0(double& x) { x = x > 0 ? 1 : 0; }
void aF_logLike(double& x)    { x = x > 0 ? log(1 + x) : -log(1 - x); }


// RESEAU - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class NeuralNetwork {
private:
    vector<unsigned short> sizes; // forme du reseau de neurones
    unsigned short layer_number;  // nombre de couche (sizes.size())
    void (*activation_function)(double&); // fonction d'activation
    bool last_activation; // derniere couche affectee ?
    string save_path = "Coefs.txt";

    unsigned short len; // nombre de coefficients
    vector<vector<vector<unsigned short>>> ind_w; // tableaux d'indices de w et b
    vector<vector<unsigned short>>         ind_b;
    vector<vector<double>> a; // valeurs de chaque couche

    vector<double> Grad;
    double (*Loss)(NeuralNetwork&, vector<vector<vector<double>>>& loss_data);
    vector<vector<vector<double>>> loss_data; // ensemble( de listes( de vecteurs ))
    double gamma, beta, beta2, eps, dx;

public:
    vector<double> Coefs;

    // init
    NeuralNetwork(vector<unsigned short> sizes, 
                  void (*activation_function)(double&) = aF_sigmoid,
                  bool last_activation = false, 
                  vector<double> Coefs = vector<double>(),
                  int random_seed      = 0 ): 

                  sizes(sizes), layer_number(sizes.size()), activation_function(activation_function),
                  last_activation(last_activation), save_path("Coefs.txt") {

        
        // tableaux d'indices de w et b
        len = 0;
        vector<unsigned short> temp0{0}; ind_b.push_back(temp0);
        vector<vector<unsigned short>> temp1{temp0}; ind_w.push_back(temp1); // skip l'indice 0
        for (int i = 1; i < layer_number; i++) {
            vector<vector<unsigned short>> wi;
            vector<unsigned short>         bi;
            for (int j = 0; j < sizes[i]; j++) {
                vector<unsigned short> wij;
                for (int k = 0; k < sizes[i-1]; k++) {
                    wij.push_back(len++); }
                wi.push_back(wij);
            }
            for (int j = 0; j < sizes[i]; j++) {
                bi.push_back(len++);
            }
            ind_w.push_back(wi);
            ind_b.push_back(bi);
        }

        // tableau des coefficients
        if (Coefs.empty()) {
            default_random_engine generator(random_seed);
            normal_distribution<double> distribution(0.0, 1.0);
            double nbr;
            for (int i = 0; i < len; i++) {
                nbr = distribution(generator);
                this->Coefs.push_back(nbr);
            }
        } else {this->Coefs = Coefs;}

        // initialisation de a et Grad
        for (int i = 0; i < layer_number; i++) {
            a.push_back(vector<double>(sizes[i],0)); }
        Grad = vector<double>(len,0);
    }

    // sauvegarder ou exporter des coefficients
    void write_Coefs(string path = "") {
        if (path == "") { path = save_path; }
        fstream file(path, ios::trunc);
        file.open(path);
        for (int i = 0; i < len-1; i++) {
            file << Coefs[i] << "\n"; }
        file << Coefs[len-1];
        file.close(); }
    void import_Coefs(string path = "") {
        if (path == "") { path = save_path; }
        fstream file(path, ios::in);
        string line;
        int i = 0;
        while (file.good()) {
            file >> line;
            Coefs[i++] = stod(line); }
        file.close(); }

    // evalue le gradient
    void eval_Grad(double &loss0) {
        for (int i = 0; i < len; i++) {
            Coefs[i] += dx;
            Grad[i]   = (Loss(*this, loss_data)-loss0)/dx;
            Coefs[i] -= dx;
        }
    }

    // appel : NN.eval(x)
    vector<double> eval(vector<double> &x) {
        a[0] = x;
        for (int i = 1; i < layer_number; i++) {
            for (int j = 0; j < sizes[i]; j++) {
                a[i][j] = 0;
                for (int k = 0; k < sizes[i-1]; k++) {
                    a[i][j] += Coefs[ind_w[i][j][k]] * a[i-1][k]; }
                a[i][j] += Coefs[ind_b[i][j]];
                if (last_activation || i != layer_number -1) {
                    activation_function(a[i][j]);
                }
            }
        }
        return a[layer_number-1];
    }

    // OPTIMISEURS  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    //// init
    void basic_init(double (*Loss)(NeuralNetwork&, vector<vector<vector<double>>> &loss_data),
                    vector<vector<vector<double>>> &loss_data,
                    double gamma = 1.E-5, double dx = 1.E-5) {
        this->Loss  = Loss;
        this->loss_data = loss_data;
        this->gamma = gamma;
        this->dx    = dx; }
    void momentum_init(double (*Loss)(NeuralNetwork&, vector<vector<vector<double>>> &loss_data),
                      vector<vector<vector<double>>> &loss_data,
                       double gamma = 1.E-4, double beta = 0.999, double dx = 1.E-5) {
        this->Loss  = Loss;
        this->loss_data = loss_data;
        this->gamma = gamma;
        this->beta  = beta;
        this->dx    = dx; }
    void AdaGrad_init(double (*Loss)(NeuralNetwork&, vector<vector<vector<double>>> &loss_data),
                      vector<vector<vector<double>>> &loss_data,
                      double gamma = 1.E-1, double eps = 1.E-5, double dx = 1.E-5) {
        this->Loss  = Loss;
        this->loss_data = loss_data;
        this->gamma = gamma;
        this->eps   = eps;
        this->dx    = dx; }
    void RMSprop_init(double (*Loss)(NeuralNetwork&, vector<vector<vector<double>>> &loss_data),
                      vector<vector<vector<double>>> &loss_data,
                      double gamma = 1.E-3, double beta = 0.97, double eps = 1.E-5, double dx = 1.E-5) {
        this->Loss  = Loss;
        this->loss_data = loss_data;
        this->gamma = gamma;
        this->beta  = beta;
        this->eps   = eps;
        this->dx    = dx; }
    void Adam_init(double (*Loss)(NeuralNetwork&, vector<vector<vector<double>>> &loss_data),
                      vector<vector<vector<double>>> &loss_data,
                   double gamma = 1.E-2, double beta = 0.99, double beta2 = 0.98, double eps = 1.E-5, double dx = 1.E-5) {
        this->Loss  = Loss;
        this->loss_data = loss_data;
        this->gamma = gamma;
        this->beta  = beta;
        this->beta2 = beta2;
        this->eps   = eps;
        this->dx    = dx; }

    //// algorithmes

    void basic(int iterations) {
        double loss0 = Loss(*this, loss_data);
        double loss1i;
        double best_loss = loss0;
        vector<double> best_Coefs = Coefs;

        for (int n = 1; n <= iterations; n++) {
            eval_Grad(loss0);
            for (int i = 0; i < len; i++) {
                Coefs[i] -= gamma*Grad[i]; 
            }
            loss0 = Loss(*this, loss_data);
            if (loss0 < best_loss) { best_loss = loss0; best_Coefs = Coefs; }
            if (n%10==0) { cout << n << " :\t" << setprecision(20) << loss0 << endl;}
        }
        Coefs = best_Coefs;
        write_Coefs();
    }

    void momentum(int iterations) {
        double loss0 = Loss(*this, loss_data);
        double best_loss = loss0;
        vector<double> best_Coefs = Coefs;

        vector<double> V(len,0);

        for (int n = 1; n <= iterations; n++) {
            eval_Grad(loss0);
            for (int i = 0; i < len; i++) {
                V[i]      = beta*V[i] + (1-beta)*Grad[i];
                Coefs[i] -= gamma*V[i]; 
            }
            loss0 = Loss(*this, loss_data);
            if (loss0 < best_loss) { best_loss = loss0; best_Coefs = Coefs; }
            if (n%10==0) { cout << n << " :\t" << setprecision(20) << loss0 << endl;}
        }
        Coefs = best_Coefs;
        write_Coefs();
    }

    void AdaGrad(int iterations) {
        double loss0 = Loss(*this, loss_data);
        double best_loss = loss0;
        vector<double> best_Coefs = Coefs;

        vector<double> G(len,0);

        for (int n = 1; n <= iterations; n++) {
            eval_Grad(loss0);
            for (int i = 0; i < len; i++) { 
                G[i]     += pow(Grad[i],2);
                Coefs[i] -= gamma/(sqrt(G[i])+eps)*Grad[i];
            }
            loss0 = Loss(*this, loss_data);
            if (loss0 < best_loss) { best_loss = loss0; best_Coefs = Coefs; }
            if (n%10==0) { cout << n << " :\t" << setprecision(20) << loss0 << endl;}
        }
        Coefs = best_Coefs;
        write_Coefs();
    }

    void RMSprop(int iterations) {
        double loss0 = Loss(*this, loss_data);
        double best_loss = loss0;
        vector<double> best_Coefs = Coefs;

        vector<double> V(len,0);

        for (int n = 1; n <= iterations; n++) {
            eval_Grad(loss0);
            for (int i = 0; i < len; i++) { 
                V[i]      = beta*V[i] + (1-beta)*pow(Grad[i],2);
                Coefs[i] -= gamma/(sqrt(V[i])+eps)*Grad[i];
            }
            loss0 = Loss(*this, loss_data);
            if (loss0 < best_loss) { best_loss = loss0; best_Coefs = Coefs; }
            if (n%10==0) { cout << n << " :\t" << setprecision(20) << loss0 << endl;}
        }
        Coefs = best_Coefs;
        write_Coefs();
    }

    void Adam(int iterations) {
        double loss0 = Loss(*this, loss_data);
        double best_loss = loss0;
        vector<double> best_Coefs = Coefs;

        vector<double> M(len,0), V(len,0);
        double m_hat, v_hat;
        for (int n = 1; n <= iterations; n++) {
            eval_Grad(loss0);
            for (int i = 0; i < len; i++) { 
                M[i]      = beta *M[i] + (1-beta )*Grad[i];
                V[i]      = beta2*V[i] + (1-beta2)*pow(Grad[i],2);
                m_hat     = M[i]/(1-pow(beta ,n));
                v_hat     = V[i]/(1-pow(beta2,n));
                Coefs[i] -= gamma*m_hat/(sqrt(v_hat)+eps);
            }
            loss0 = Loss(*this, loss_data);
            if (loss0 < best_loss) { best_loss = loss0; best_Coefs = Coefs; }
            if (n%10==0) { cout << n << " :\t" << setprecision(20) << loss0 << endl;}
            //cout << setprecision(20) << Grad[10] << endl;

        }
        Coefs = best_Coefs;
        write_Coefs();
    }
};


// LOSS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

double MSE(NeuralNetwork& NN, 
           vector<vector<vector<double>>> &loss_data) {
    // loss_data de la forme {Xref, Yref}
    vector<vector<double>> &Xref = loss_data[0];
    vector<vector<double>> &Yref = loss_data[1];
    double sum = 0;
    int N = Xref.size(), nf = Yref[0].size();
    for (int i = 0; i < N; i++) {
        vector<double> Yi = NN.eval(Xref[i]);
        for (int j = 0; j < nf; j++) {
            sum += pow( Yref[i][j] - Yi[j] ,2); }
    }
    return sum/(N*nf);
}

double PhyLoss(NeuralNetwork& NN, 
           vector<vector<vector<double>>> &loss_data) {
    // loss_data de la forme {Tphy}

    double m = 1, f = 1, k = 10;
    double x0 = 3, v0 = 0;
    double dt = 1.E-5;

    vector<vector<double>> &Tphy = loss_data[0];
    double phy = 0, ref = 0;
    int N = Tphy.size();
    cout << "test" << endl;
    for (int i = 0; i < N; i++) {
        vector<double>  t_a {Tphy[i][0]+dt};
        vector<double> &t_b = Tphy[i];
        vector<double>  t_c {Tphy[i][0]-dt};
        double eval_a = NN.eval(t_a)[0];
        double eval_b = NN.eval(t_b)[0];
        double eval_c = NN.eval(t_c)[0];
        phy += pow(m*(eval_a - 2*eval_b + eval_c)/dt/dt
                 + f*(eval_a -   eval_b         )/dt
                 + k*(           eval_b         ), 2);
    }
    vector<double> a{0}, b{dt};
    ref += pow(NN.eval(a)[0] - x0, 2);
    ref += pow((NN.eval(b)[0] - NN.eval(a)[0])/dt - v0, 2);
    return ref/2 + phy/(10*N);
}


// AUTRES TRUCS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

vector<vector<vector<double>>> A2data() { // loss_data
    vector<vector<double>> X, Y;
    double x, y;
    int N = 30;
    for (int i = 0; i < N; i++) {
        x = -10 + 20*i/(1.*N);
        y = ( cosh(x) - x*x + x )/5 - 10;
        y = 10 + y - y*y/sqrt(y*y + 5);
        X.push_back(vector<double>{x});
        Y.push_back(vector<double>{y});
    }
    vector<vector<vector<double>>> XY {X,Y};
    return XY;
}

vector<vector<vector<double>>> A3data() {
    vector<vector<double>> T;
    double t;
    int N = 20;
    for (int i = 1; i <= N; i++) {
        t = 6*i/(1.*N);
        T.push_back(vector<double>{t});
    }
    return vector<vector<vector<double>>>{T};
}

// MAIN - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void A2() {
    cout << "init" << endl;
    vector<unsigned short> sizes = {1,10,10,1};
    NeuralNetwork NN(sizes);
    vector<vector<vector<double>>> Xref_Yref = A2data();
    NN.Adam_init(MSE, Xref_Yref);
    cout << "optimizing" << endl;
    NN.Adam(2000);
    cout << "finito" << endl;
}

void A3() {
    cout << "init" << endl;
    vector<unsigned short> sizes = {1,10,10,1};
    NeuralNetwork NN(sizes);
    vector<vector<vector<double>>> Tphy = A3data();
    //for (int i = 0; i<30; i++) { cout << Tphy[0][i][0] << endl;} ????????????????????????????????????
    cout << "test" << endl;
    NN.Adam_init(MSE, Tphy);
    cout << "optimizing" << endl;
    NN.Adam(1000);
    cout << "finito" << endl;
}

int main() {
    A3();
    return 0;
}

// g++ .\A2_ReseauCPP.cpp -o A2_ReseauCPP; .\A2_ReseauCPP.exe 