import numpy as np
from scipy import linalg as LA
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold
import time
import itertools

start_time = time.time()

def loss(X, Y, G, beta, gamma, corr_func, corr_thresh, lambda_):
    return np.real(rss(X, Y, beta) + l1_penalty(beta, lambda_) + fusion_penalty(G, beta, gamma, corr_func, corr_thresh))

def rss(X, Y, beta):
    rss = 0
    # For each drug
    for k in range(np.size(Y, 1)):
        # Compute (y_k - XB_k)
        y_k = Y[:,k]
        beta_k = beta[:,k]
        y_delta = y_k - np.matmul(X, beta_k)
        # Take the transpose
        # Multiply the transpose with (y_k - XB_k)
        rss_k = np.matmul(y_delta, y_delta.T)
        rss += rss_k

    return(rss)

def fusion_penalty(G, beta, gamma, corr_func, corr_thresh):
    penalty = 0

    for m in range(0, np.size(G, 0)): # Iterate through rows
        for l in range(m, np.size(G, 1)): # Iterate through cols, skipping "edges" that already occured
            r_ml = G[m][l]
            abs_sum = 0
            for j in range(0, np.size(beta, 0)): # Iterate through rows
                abs_sum += abs(beta[j][m] - np.sign(r_ml) * beta[j][l])
            f_r_ml = correlation_function(r_ml, corr_func, corr_thresh)
            penalty += f_r_ml * abs_sum
    
    # Multiply penalty by gamma
    penalty = gamma * penalty
    
    return(penalty)

def l1_penalty(beta, lambda_):
    return np.sum(lambda_*np.absolute(np.array(beta)))

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
    if corr_func == 'absolute':
        f_r_ml = absolute_correlation(r_ml)
    elif corr_func == 'squared':
        f_r_ml = squared_correlation(r_ml)
    elif corr_func == 'thresholded':
        f_r_ml = thresholded_correlation(r_ml, corr_thresh)
    else:
        raise ValueError("Unrecognized correlation function. Please correct to \"absolute\", \"squared\", or \"thresholded\"")
    return f_r_ml

def get_mu(J, K, E_size, epsilon):
    D = 1 / 2 * J * (K + E_size)
    mu_result = epsilon / (2 * D)
    return mu_result

def get_C(lambda_, gamma, K, E_size, G, corr_func, corr_thresh):
    I = np.identity(K)
    H = np.zeros((K, E_size))
    for k in range (0, K):
        e = 0
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
    return np.concatenate((lambda_ * I, gamma * H), axis=1)

def get_L_U(X, G, K, mu, lambda_, gamma, corr_func, corr_thresh):
    #eigvals = LA.eigvals(np.matmul(X.T, X))
    #lambda_max = max(eigvals)
    lambda_max = 877783.158 # Value printed from previous run
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

def get_A_star(W_t, C, mu):
    if mu == 0:
        raise ValueError("Parameter mu cannot be zero, it would cause division by zero.")
    inner_matrix = np.matmul(W_t, C) / mu
    # with np.nditer(inner_matrix, op_flags=['readwrite']) as it: # TODO: make this faster
    #     for x in it:
    #         if x >= 1:
    #             x[...] = 1
    #         elif x <= -1:
    #             x[...] = -1
    inner_matrix = np.where(inner_matrix <= -1, -1, inner_matrix)
    inner_matrix = np.where(inner_matrix >= 1, 1, inner_matrix)
    return inner_matrix

def proximal_gradient_descent(G, X, Y, lambda_, gamma, epsilon, max_iter, corr_func, corr_thresh=None):
    split_index = int(0.9 * len(X))
    X_90, X_10 = X[:split_index], X[split_index:]
    Y_90, Y_10 = Y[:split_index], Y[split_index:]

    J = np.size(X, 1)
    K = np.size(Y, 1)
    E_size = K**2
    mu = get_mu(J, K, E_size, epsilon)
    C = get_C(lambda_, gamma, K, E_size, G, corr_func, corr_thresh)
    L_U = get_L_U(X_90, G, K, mu, lambda_, gamma, corr_func, corr_thresh)
    if L_U == 0:
        raise ValueError("L_U cannot be zero, it would cause dividion by zero.")  
    W_t = np.zeros((J, K))
    B_t = None
    best_B_t = None # Store weights for which had best val MSE
    best_val_MSE = None # Store best val MSE
    best_iteration = None # Iteration where found lowest val MSE
    t = 0
    while t < max_iter:
        print("Iteration number: " + str(t) + ", Time: " + str(time.time() - start_time) + " seconds")
        A_star = get_A_star(W_t, C, mu)
        delta_f_tilde = np.add(np.matmul(X_90.T, np.subtract(np.matmul(X_90, W_t), Y_90)), np.matmul(A_star, np.transpose(C)))
        B_t = np.subtract(W_t, delta_f_tilde / L_U)
        
        # Validation
        Y_90_preds = np.real(np.matmul(X_90, B_t))
        train_MSE = mse(Y_90, Y_90_preds)
        Y_10_preds = np.real(np.matmul(X_10, B_t))
        val_MSE = mse(Y_10, Y_10_preds)

        if t == 0:
            Z_t = np.zeros(delta_f_tilde.shape)
            
            # Update best val MSE and weights
            best_val_MSE = val_MSE
            best_B_t = B_t
        else:
            Z_t = np.add(Z_t, (-1 / L_U) * (t + 1) / 2 * delta_f_tilde)

            # Update best val MSE and weights
            if val_MSE < best_val_MSE:      
                best_val_MSE = val_MSE
                best_B_t = B_t

        W_t = np.add((t + 1) / (t + 3) * B_t, 2 / (t + 3) * Z_t)
        t = t + 1
        print("Train MSE: " + str(train_MSE))
        print("Val MSE: " + str(val_MSE))
        print("Best val MSE: " + str(best_val_MSE))
        print("")

    return B_t

def cross_validate(X, Y, G, lambda_, gamma, epsilon, max_iter, corr_func, corr_thresh=None, num_folds=5):
    """
    num_folds: number of folds to cross-validate against
    """
    print("")
    
    cv_scores = []
    
    # Split data into training/holdout sets
    kf = KFold(n_splits=num_folds, shuffle=True)
    kf.get_n_splits(X)
    
    # Keep track of the training and validation scores
    train_scores = []
    val_scores = []
    
    # Iterate over folds, using k-1 folds for training
    # and the k-th fold for validation
    f = 1
    for train_index, test_index in kf.split(X):
        # Training data
        CV_X = X[train_index,:]
        CV_Y = Y[train_index]
        
        # Holdout data
        holdout_X = X[test_index,:]
        holdout_Y = Y[test_index]
        
        # Fit model to training sample
        beta = proximal_gradient_descent(
            G=G, X=CV_X, Y=CV_Y, lambda_=lambda_, gamma=gamma, epsilon=epsilon, corr_func=corr_func, max_iter=max_iter, corr_thresh=corr_thresh
        )

        # Calculate training error
        train_preds = np.real(np.matmul(CV_X, beta))
        train_mse = mse(CV_Y, train_preds)
        train_scores.append(train_mse)
        
        # Calculate holdout error
        fold_preds = np.real(np.matmul(holdout_X, beta))
        fold_mse = mse(holdout_Y, fold_preds)
        val_scores.append(fold_mse)
        print("Fold: {}. Train MSE: {}. Validation MSE: {}".format(f, train_mse, fold_mse))
        f += 1
    
    # Get average training and validation score
    mean_train_score = np.mean(train_scores)
    mean_val_score = np.mean(val_scores)
    print("\nAVERAGE TRAIN MSE: {}".format(mean_train_score))
    print("AVERAGE VALIDATION MSE: {}".format(mean_val_score))
    print("")
    
    return mean_train_score, mean_val_score

def grid_search_cv(X, Y, G, parameters, num_folds):

    best_params = None
    best_train_score = None
    best_val_score = None

    # Iterate through each candidate
    round_num = 1
    keys, values = zip(*parameters.items())
    for v in itertools.product(*values):
        candidate_params = dict(zip(keys, v))
        print("\n================================ GRID SEARCH CV - ROUND {} ================================".format(round_num))
        print("\nHyperparameters: {}".format(candidate_params))

        mean_train_score, mean_val_score = cross_validate(
            X=X, Y=Y, G=G,
            lambda_=candidate_params['lambda_'], 
            gamma=candidate_params['gamma'], 
            epsilon=candidate_params['epsilon'], 
            max_iter=candidate_params['max_iter'], 
            corr_func=candidate_params['corr_func'], 
            corr_thresh=candidate_params['corr_thresh'],
            num_folds=num_folds
        )

        # Update best candidate model
        if best_val_score == None or mean_val_score < best_val_score:
            best_params = candidate_params
            best_train_score = mean_train_score
            best_val_score = mean_val_score

        round_num += 1

    print("\n================================ GRID SEARCH CV - FINAL RESULTS ================================".format(round_num))
    print("\nBest hyperparameters: {}".format(best_params))
    print("Train MSE: {}".format(best_train_score))
    print("Validation MSE: {}".format(best_val_score))

    return best_params, best_train_score, best_val_score

data_path = '../Data/'
results_path = '../Results/'

#  ===================== TRAINING SECTION ========================
print("Retrieving data ....")
drug_names = pd.read_csv(data_path + 'drug_drug_similarity.csv',index_col=0, header=None, low_memory=False).T.set_index('drug').apply(pd.to_numeric)
train_x = pd.read_csv(data_path + 'gdsc_expr_postCB(normalized).csv', index_col=0, header=None, low_memory=False).T.set_index('cell line id').apply(pd.to_numeric)#.iloc[:,0:500]
train_y = pd.read_csv(data_path + 'gdsc_dr_lnIC50.csv', index_col=0, header=None, low_memory=False).T.set_index('cell line id').apply(pd.to_numeric)#.iloc[:,]

#select 8 drugs which only exists in the drug-drug similarity matrix
train_y = train_y.filter(drug_names)

# If a drug's DR is NaN, set it to the mean of the cell lines' DR for that drug
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
train_y = pd.DataFrame(data=imp.fit_transform(train_y), index=train_y.index, columns=train_y.columns)

correlation_matrix = drug_names.values
X = train_x.values
Y = train_y.values
print("Data ready to train")

# Hyperarameters to test for grid search
parameters = {
    'lambda_':[10],
    'gamma':[10],
    'epsilon':[1],
    'max_iter':[50],
    'corr_func':['absolute'],
    'corr_thresh':[1]
}
best_params, best_train_score, best_val_score = grid_search_cv(X=X, Y=Y, G=correlation_matrix, parameters=parameters, num_folds=5)

print("\nTotal runtime was: " + str(time.time() - start_time) + " seconds")