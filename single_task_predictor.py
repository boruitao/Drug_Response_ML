import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import ElasticNetCV
from sklearn.datasets import make_regression

################# SINGLE TASK DRUG RESPONSE PREDICTOR ################# 

# Takes two numpy arrays as input: y_actual, y_predicted
# Returns two sample groups:
#   The first contains a list of predicted drug response values for which the patient's actual response was 0
#   The second contains a list of predicted drug response values for which the patient's actual response was 1
def get_t_test_groups(y_actual, y_predicted):
    
    # Ensure the sizes are the same
    try:
        assert len(y_actual) == len(y_predicted)
    except AssertionError:
        print("AssertionError: size of y_actual and y_predicted must be the same")

    # Construct the two sample groups
    drug_responses_0 = []
    drug_responses_1 = []
    for i in range(len(y_actual)):
        if y_actual[i] == 0:
            print("0 found!")
            drug_responses_0.append(y_predicted[i])
        elif y_actual[i] == 1:
            print("1 found!")
            drug_responses_1.append(y_predicted[i])

    return drug_responses_0, drug_responses_1

# Path to datasets
data_path = '../Data/'

# Load training and test set
x_train = pd.read_csv(data_path + 'gdsc_expr_postCB.csv', index_col=0, header=None).T.set_index('cell line id')
y_train = pd.read_csv(data_path + 'gdsc_dr_lnIC50.csv', index_col=0, header=None).T.set_index('cell line id')
x_test = pd.read_csv(data_path + 'tcga_expr_postCB.csv', index_col=0, header=None).T.set_index('patient id')
y_test = pd.read_csv(data_path + 'tcga_dr.csv', index_col=0, header=None).T.set_index('patient id')

# Normalize data for mean 0 and standard deviation of 1
ss = StandardScaler()
x_train = pd.DataFrame(ss.fit_transform(x_train), index = x_train.index, columns = x_train.columns)
y_train = pd.DataFrame(ss.fit_transform(y_train), index = y_train.index, columns = y_train.columns)
x_test = pd.DataFrame(ss.fit_transform(x_test), index = x_test.index, columns = x_test.columns)

# Replace NaN values by the mean
#x_train.fillna(x_train.mean(), inplace=True)
#x_test.fillna(x_test.mean(), inplace=True)

# Verify axes match
#compare_column_headers(x_train, x_test)
#compare_column_headers(y_train, y_test)
#compare_row_headers(x_train, y_train)
#compare_row_headers(x_test, y_test)

# T_TEST SAMPLE GROUP TEST
#y_test_actual = pd.DataFrame(0, index=y_test[['bicalutamide']].index, columns=y_test[['bicalutamide']].columns).values
#y_test_actual = np.random.randint(2, size=20) # Fill with either 0 or 1 randomly
#y_test_predicted = np.random.rand(len(y_test_actual))
#drug_responses_0, drug_responses_1 = get_t_test_groups(y_test_actual, y_test_predicted)
#System.exit(0)

y_test_prediction = pd.DataFrame(index=y_test.index, columns=y_test.columns)

# Predict the response for each drug individually
for drug in y_train:

    # Keep only one drug column
    y_train_single = y_train[[drug]]

    # Drop cell line ids in y_train where drug response is NaN
    y_train_single = y_train_single.dropna()
    non_null_ids = y_train_single.index
    
    # Drop cell line ids in x_train where drug response is NaN
    x_train_single = x_train[x_train.index.isin(non_null_ids)]

    # Create elastic net model with five-fold cross validation 
    regr = ElasticNetCV(cv=5, random_state=0)
    regr.fit(x_train_single.values, np.ravel(y_train_single.values))

    # Predict the y_test drug response
    y_test_prediction_single = predict(x_test)
    y_test_actual_single = y_test_binary[[drug]]
    drug_responses_0, drug_responses_1 = get_t_test_groups(y_test_actual_single, y_test_prediction_single)
    # TO-DO: Call the t-test on drug_responses_0 and drug_response_1

# Cross validation: used to select hyperparameters
#   Do it with diff values of alpha
#   Choose a loss function to choose the alpha that minimizes loss
#       Ex: euclidean norm, cosine similarity....
#   Find alpha with "path" --> elasticnet CV, lasso CV
#   Scikit has modules for cross validation and hyperparam selection
#   If multiple hyperparams, can have a LOT of possibilites --> scikit learn help save time
# Lasso
#   Lasso minimize 1/2 || y - x ||
#   L1: minimize absolute values of some of weights --> imposes sparsity
# Elastic
#   Has extra L2 norm of w

# Normalization of original data
#   Figure data distribution
#   
# Improve code
#   Pandas, generalizability of data loading
#   Hyperparameter tuning
#   Normalization
#   T-test: scipy
#   Can report: cross validation accuracy (check for overfitting/underfitting)
#   Can try nonlinear models: e.g. Support vector regression

# Sensitive: complete response, partial response
# Resistant: stable disease, progressive disease

# Spend time learning multi-task methods

# Profile compute canada: find what is accessible to us, etc...
#       Takes time when requesting resources: in case need for next semester
#       Access GPUs: good for neural networks

# Right things down for report immediately: so that he can give us feedback