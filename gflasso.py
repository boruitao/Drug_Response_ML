import numpy as np
from numpy import linalg as LA
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error AS mse


def absolute_correlation(r_ml):
    return abs(r_ml)

def squared_correlation(r_ml):
    return r_ml**2

def thresholded_correlation(r_ml, corr_thresh):
    if r_ml > corr_thresh:
        return 1
    else:
        return 0

def correlation_function(r_ml, corr_func, corr_thresh):
    f_r_ml = None
    if self.corr_func == 'absolute':
        f_r_ml = self.absolute_correlation(r_ml)
    elif self.corr_func == 'squared':
        f_r_ml = self.squared_correlation(r_ml)
    elif self.corr_func == 'thresholded':
        f_r_ml = self.thresholded_correlation(r_ml, corr_thresh)
    else:
        raise ValueError("Unrecognized correlation function. Please correct to \"absolute\", \"squared\", or \"thresholded\"")
    return f_r_ml

def mu(J, K, E_size, epsilon):
    D = 1 / 2 * J * (K + E_size)
    mu = epsilon / (2 * D)
    return mu

def C(lambda_, gamma, K, E_size, G, corr_func, corr_thresh):
    I = np.identity(K)
    H = np.zeros((K, E_size))
    e = 0
    for k in range (0, K):
        for m in range(0, K):
            for l in range(0, K):
                r_ml = G[m][l]
                f_r_ml = correlation_function(r_ml, corr_func, corr_thresh)
                if k == m:
                    H[k, e] = f_r_ml
                elif k == l:
                    H[k, e] = -np.sign(r_ml) * f_r_ml
                else:
                    H[k, e] = 0
                e += 1
    #return lambda_ * I, gammma * H # Return two matrices: not sure if this is what C is supposed to be
    return np.concatenate((ambda_ * I, gammma * H), axis=1)

def L_U(X, G, K, mu, lambda_, gamma, corr_func, corr_thresh):
    w, v = LA.eig(np.matmul(X.T, X))
    lambda_max = max(w)
    d_k_max = 0
    for k in range(0, K):
        d_k = 0
        for m in range(0, K):
            r_km = G[k][m]
            f_r_km = correlation_function(r_km, corr_func, corr_thresh)
            d_k += f_r_km**2
        if d_k > d_k_max:
            d_k_max = d_k
    L_U = lambda_max + (lambda_**2 + 2 * gamma**2 * d_k_max) / mu
    return L_U

def A_star(W_t, C, mu):
    if mu == 0:
        raise ValueError("Parameter mu cannot be zero, it would cause division by zero.")
    x = np.matmul(W_t, C) / mu # TO-DO: verify what C is
    if x > -1 and x < -1:
        return x
    elif x >= 1:
        return 1
    else:
        return -1

def proximal_gradient_descent(G, X, Y, J, K, lambda_, gamma, epsilon, corr_func, corr_thresh=None, iterations=100):
    E_size = K**2
    mu = mu(J, K, E_size, epsilon)
    C = C(K, G)
    L_U = L_U(X, G, K, mu, lambda_, gamma)
    if L_U == 0:
        raise ValueError("L_U cannot be zero, it would cause dividion by zero.")
    
    W_t = np.zeros((J, K))
    B_t = None
    cost_history = np.zeros(iterations) # Store sequence of costs at each time t
    B_t_history = np.zeros(iterations) # Store sequence of weights at each time t (B_t)
    t = 0
    while t < iterations: # Instead of checking for convergence, do a set amount of iterations
        A_star = A_star(W_t, C, mu)
        #delta_f_tilde = np.add(np.matmul(X.T, np.subtract(np.matmul(X, W_t), Y)), A_star * C)) # TO-DO: verify what C is
        delta_f_tilde = np.add(np.matmul(X.T, np.subtract(np.matmul(X, W_t), Y)), A_star * np.transpose(C)))
        B_t = np.subtract(W_t, delta_f_tilde / L_U)
        B_t_history[t] = B_t
        if t == 0:
            Z_t = np.zeros(delta_f_tilde.shape)
        else:
            Z_t = np.add(Z_t, (-1 / L_U) * (t + 1) / 2 * delta_f_tilde)
        W_t = np.add((t + 1) / (t + 3) * B_t, 2 / (t + 3) * Z_t)
        
        # Calculate cost (MSE)
        Y_pred = np.matmul(X, W_t)
        cost = mse(Y, Y_pred)
        cost_history[t] = cost

        t = t + 1
    return B_t, B_t_history, cost_history

#  ===================== TRAINING SECTION ========================
print("Retrieving data ....")
data_path = '../Data/'
results_path = '../Results/'
drug_cids = pd.read_csv(data_path + 'drug_drug_similarity.csv',index_col=0, header=None, low_memory=False).T.set_index('drug').apply(pd.to_numeric)
train_x = pd.read_csv(data_path + 'gdsc_expr_postCB(normalized).csv', index_col=0, header=None, low_memory=False).T.set_index('cell line id').apply(pd.to_numeric)#.iloc[0:10:,0:5]
train_y = pd.read_csv(data_path + 'gdsc_dr_lnIC50.csv', index_col=0, header=None, low_memory=False).T.set_index('cell line id').apply(pd.to_numeric)#.iloc[0:10:,]

#select 8 drugs which only exists in the drug-drug similarity matrix
train_y = train_y.filter(drug_cids)

# If a drug's DR is NaN, set it to the mean of the cell lines' DR for that drug
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
train_y = pd.DataFrame(data=imp.fit_transform(train_y), index=train_y.index, columns=train_y.columns)

correlation_matrix = drug_cids.values
X = train_x.values
Y = train_y.values
print("Data ready to train")