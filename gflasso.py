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

# TO-DO: Bijan
def A_star(W_t, C, mu):
    return 0

# TO-DO: Bijan
def proximal_gradient_descent(G, X, Y, J, K, lambda_, gamma):
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
