import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

# Example loss function used in article (MAPE)
# [TO-DO]: replace this function by the gflasso loss function
def mean_absolute_percentage_error(y_pred, y_true, sample_weights=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert len(y_true) == len(y_pred)
    
    if np.any(y_true==0):
        print("Found zeroes in y_true. MAPE undefined. Removing from set...")
        idx = np.where(y_true==0)
        y_true = np.delete(y_true, idx)
        y_pred = np.delete(y_pred, idx)
        if type(sample_weights) != type(None):
            sample_weights = np.array(sample_weights)
            sample_weights = np.delete(sample_weights, idx)
        
    if type(sample_weights) == type(None):
        return(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    else:
        sample_weights = np.array(sample_weights)
        assert len(sample_weights) == len(y_true)
        return(100/sum(sample_weights)*np.dot(
                sample_weights, (np.abs((y_true - y_pred) / y_true))
        ))

def absolute_correlation(r_ml):
        return abs(r_ml)

def squared_correlation(r_ml):
    return r_ml**2

def thresholded_correlation(r_ml):
    if r_ml > correlation_threshold:
        return 1
    else:
        return 0

def GFLasso_loss(y_pred, y_true, sample_weights=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert len(y_true) == len(y_pred)
    
    if np.any(y_true==0):
        print("Found zeroes in y_true. MAPE undefined. Removing from set...")
        idx = np.where(y_true==0)
        y_true = np.delete(y_true, idx)
        y_pred = np.delete(y_pred, idx)
        if type(sample_weights) != type(None):
            sample_weights = np.array(sample_weights)
            sample_weights = np.delete(sample_weights, idx)
        
    if type(sample_weights) == type(None):
        return(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    else:
        sample_weights = np.array(sample_weights)
        assert len(sample_weights) == len(y_true)
        return(100/sum(sample_weights)*np.dot(
                sample_weights, (np.abs((y_true - y_pred) / y_true))
        ))

class GFLasso:
    """
    GFLasso model: Y = XB, fit by minimizing (RSS + L1 penalty + Fusion penalty)
    
    Y: (n x k) matrix of true Y values
        n: number of samples
        k: number of tasks
    
    X: (n x p) matrix of true X values
        n: number of samples
        p: number of features
    
    beta_init: (p x k) matrix of weights
        p: number of features
        k: number of tasks

    lambda: used in l1 penalty

    gamma: used in fusion penalty

    correlation_function: the correlation function used for the fusion penalty
    correlation_threshold: the threshold used if correlation_function='threshold'
    """
    def __init__(self, 
                 X=None, Y=None, beta_init=None,
                 lambda_=0, gamma=0,
                 correlation_function=absolute_correlation, correlation_threshold=0.5):
        self.lambda_ = lambda_
        self.beta = None
        self.beta_init = beta_init
        
        self.X = X
        self.Y = Y

        self.correlation_function = correlation_function
    
    def predict(self, X):
        prediction = np.matmul(X, self.beta)
        return(prediction)
    
    def loss(self, beta):
        self.beta = beta
        loss = (self.rss() + \
               self.l1_penalty() + \
               self.fusion_penalty()
               )
        return loss

    # TO-DO
    def rss(self):
        self.beta = np.reshape(self.beta, (-1, np.size(self.beta_init, 1))) # Make beta 2D to allow calculations

        rss = 0
        # For each drug
        for k in range(np.size(self.Y, 1)):
            # Compute (y_k - XB_k)
            y_k = self.Y[:,k]
            beta_k = self.beta[:,k]
            y_delta = y_k - np.matmul(self.X, beta_k)
            # Take the transpose
            # Multiply the transpose with (y_k - XB_k)
            rss_k = np.matmul(y_delta, y_delta.T)
            rss += rss_k

        self.beta = np.reshape(self.beta, (1, -1)) # Make beta 1D again to allow minimize function to work
        return(rss)

    # TO-DO
    def fusion_penalty(self):
        error = 0

        # For each entry (r_ml) in drug-drug similarity matrix
            # For each entry r(_jm) in drug-drug similarity matrix
                # Take entry beta_jm? --> Not sure what beta_jm refers to sice beta is (genes x drugs), not (drugs x drugs)

        return(error)
    
    def l1_penalty(self):
        return np.sum(self.lambda_*np.absolute(np.array(self.beta)))
    
    def fit(self, maxiter=250):     
        # Initialize beta estimates (you may need to normalize
        # your data and choose smarter initialization values
        # depending on the shape of your loss function)
        if type(self.beta_init)==type(None):
            # set beta_init = 1 for every feature
            self.beta_init = np.ones([X.shape[1], Y.shape[1]], dtype = float)
        else: 
            # Use provided initial values
            pass
            
        if self.beta!=None and all(self.beta_init == self.beta):
            print("Model already fit once; continuing fit with more itrations.")
        res = minimize(self.loss, self.beta_init,
                       method='BFGS', options={'maxiter': 500})
        self.beta = res.x
        self.beta_init = self.beta

loss_function = mean_absolute_percentage_error # Set loss function to MAPE

# Evaluates the error found by loss_function from beta, X, and Y (true)
def objective_function(beta, X, Y):
    error = loss_function(np.matmul(X,beta), Y)
    return(error)

# ===================== CREATE DATASETS ========================

# p = 10 features
# k = 5 tasks
# n = 20 samples

# Generate predictors
X_raw = np.random.random(20*9)
X_raw = np.reshape(X_raw, (20, 9))

# Standardize the predictors
scaler = StandardScaler().fit(X_raw)
X = scaler.transform(X_raw)

# Add an intercept column to the model.
X = np.abs(np.concatenate((np.ones((X.shape[0],1)), X), axis=1)) # (20 x 10)

# Define my "true" beta coefficients
beta = np.array([[2,6,7,3,5], # (20 x 5)
    [2,6,7,3,5],
    [2,6,7,3,5],
    [2,6,7,3,5],
    [2,6,7,3,5],
    [2,6,7,3,5],
    [2,6,7,3,5],
    [2,6,7,3,5],
    [2,6,7,3,5],
    [2,6,7,3,5]
])

# Y = Xb
Y_true = np.matmul(X,beta)
Y = Y_true

# ===================== OPTIMIZE BETA (WEIGHTS) ========================

model = GFLasso(
    X=X, Y=Y, lambda_=1, gamma=0, correlation_function = absolute_correlation
)
model.fit()
print("MODEL BETA:")
print(np.reshape(model.beta, (-1, np.size(beta, 1))))