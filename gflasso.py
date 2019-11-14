import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# TO-DO: Borui
def mu(J, K, E_size):
    return 0

# TO-DO: Borui
def C(K, G):
    return 0

# TO-DO: Borui
def L_U(X, G, K, mu, lambda_, gamma):
    return 0

def A_star(W_t, C, mu):
    if mu == 0:
        raise ValueError("Parameter mu cannot be zero, it would cause division by zero.")
    x = np.matmul(W_t, C) / mu # TO-DO: verify what C is because I don't think it's simply a matrix
    if x > -1 and x < -1:
        return x
    elif x >= 1:
        return 1
    else:
        return -1

def proximal_gradient_descent(G, X, Y, J, K, lambda_, gamma, epsilon):
    E_size = K**2
    mu = mu(J, K, E_size, epsilon)
    C = C(K, G)
    L_U = L_U(X, G, K, mu, lambda_, gamma)
    if L_U == 0:
        raise ValueError("L_U cannot be zero, it would cause dividion by zero.")
    W_0 = np.zeros((J, K))
    W_t = W_0
    t = 0
    #TO-DO: create delta_f_tilde array to hold the delta_f_tilde for each t
    converged = False
    while not converged:
        A_star = A_star(W_t, C, mu)
        # TO-DO: fix delta_f_tilde line to set the delta_f_tilde within the array for index t
        delta_f_tilde = np.add(np.matmul(X.T, np.subtract(np.matmul(X, W_t), Y)), A_star * C))
        B_t = np.subtract(W_t, delta_f_tilde / L_U)
        Z_t = np.zeros(delta_f_tilde.shape)
        for i in range(0, t + 1):
            Z_t = np.add(Z_t, (i + 1) / 2 * delta_f_tilde) # TO-DO: fix delta_f_tilde
    return 0

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