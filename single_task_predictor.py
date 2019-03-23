import numpy as np
import pandas as pd
from data_loader import compare_column_headers, compare_row_headers
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import ElasticNetCV
from sklearn.datasets import make_regression
from scipy import stats

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
        print("y_actual size: " + str(len(y_actual)))
        print("y_predicted size: " + str(len(y_predicted)))

    # Construct the two sample groups
    drug_responses_0 = []
    drug_responses_1 = []
    for i in range(len(y_actual)):
        if y_actual[i] == 0:
            drug_responses_0.append(y_predicted[i])
        elif y_actual[i] == 1:
            drug_responses_1.append(y_predicted[i])

    return drug_responses_0, drug_responses_1

# Takes in y_test and converts the four categories to 0 and 1
# "Complete response" and "Partial response" are mapped to 1
# "Stable disease" and "Clinical progressive disease" are mapped to 0
def category_to_binary(y_test):
    y_test_binary = y_test
    for drug_name in list(y_test_binary.columns.values):
        y_test_binary = y_test_binary.replace("Complete Response", 1)
        y_test_binary = y_test_binary.replace("Partial Response", 1)
        y_test_binary = y_test_binary.replace("Stable Disease", 0)
        y_test_binary = y_test_binary.replace("Clinical Progressive Disease", 0)
    return y_test_binary

# Normalizes x_train and x_test per gene
# Stores result in csv
def normalize(x_train, x_test):
    ss = StandardScaler()
    for gene in x_train:
        x_train[gene] = ss.fit_transform(x_train[[gene]].values)
    for gene in x_test:
        x_test[gene] = ss.fit_transform(x_test[[gene]].values)
    x_train.T.to_csv(data_path + 'gdsc_expr_postCB(normalized).csv')
    x_test.T.to_csv(data_path + 'tcga_expr_postCB(normalized).csv')

# Verify that the axes match
def verify_axes(x_train, y_train, x_test, y_test):
    compare_column_headers(x_train, x_test)
    compare_column_headers(y_train, y_test)
    compare_row_headers(x_train, y_train)
    compare_row_headers(x_test, y_test)

# Perform one-tailed t-test to verify the mean of resistant group is higher than the mean of sensitive group
# drug_responses_0: the resistant group
# drug_responses_1: the sensitive group
def one_tailed_t_test(drug_responses_0, drug_responses_1):
    t, p = stats.ttest_ind(drug_responses_0, drug_responses_1)

    # Correct sign
    if t >= 0:
        p = p / 2
    # Incorrect sign
    else:
        p = 1 - p / 2

    return t, p

# Path to datasets
data_path = '../Data/'

# Path to results folder
results_path = '../Results/'

# Name of model being used
model_name = 'RandomForest'

# Load training and test set
x_train = pd.read_csv(data_path + 'gdsc_expr_postCB(normalized).csv', index_col=0, header=None).T.set_index('cell line id').apply(pd.to_numeric)#.iloc[0:200, 0:200]
y_train = pd.read_csv(data_path + 'gdsc_dr_lnIC50.csv', index_col=0, header=None).T.set_index('cell line id').apply(pd.to_numeric)#.iloc[0:200, ]
x_test = pd.read_csv(data_path + 'tcga_expr_postCB(normalized).csv', index_col=0, header=None).T.set_index('patient id').apply(pd.to_numeric)#.iloc[0:200, 0:200]
y_test = pd.read_csv(data_path + 'tcga_dr.csv', index_col=0, header=None).T.set_index('patient id')#.iloc[0:200, ]
y_test_binary = category_to_binary(y_test)

# Normalize data for mean 0 and standard deviation of 1
#normalize(x_train, x_test)

# Verify axes match
#verify_axes(x_train, y_train, x_test, y_test)

# Matrix to store y_test predictions
y_test_prediction = pd.DataFrame(index=y_test.index, columns=y_test.columns)

# Matrix to store drug statistics, including t-statistic and p-value for each drug
results = y_test.describe().T.join(pd.DataFrame(index=y_test.columns, columns=['T-statistic', 'P-value']))

# Predict the response for each drug individually
for drug in y_train:

    # Keep only one drug column
    y_train_single = y_train[[drug]]

    # Drop rows in y_train where drug response is NaN, as well as corresponding x_train rows
    y_train_single = y_train_single.dropna()
    non_null_ids = y_train_single.index
    x_train_single = x_train[x_train.index.isin(non_null_ids)]

    # Create elastic net model with five-fold cross validation
    print("Fitting " + model_name + " for drug: " + drug)
    regr = RandomForestRegressor(random_state=0)
    regr.fit(x_train_single.values, np.ravel(y_train_single.values))

    # Predict y_test drug response, and insert into prediction matrix
    print("Predicting y test...")
    y_test_prediction_single = regr.predict(x_test)
    y_test_prediction[drug] = y_test_prediction_single
    
    # Perform T-test
    print("Performing t-test...")
    y_test_actual_single = y_test_binary[[drug]]
    drug_responses_0, drug_responses_1 = get_t_test_groups(y_test_actual_single.values, y_test_prediction_single)
    t, p = one_tailed_t_test(drug_responses_0, drug_responses_1)
    results.loc[drug, 'T-statistic'] = t
    results.loc[drug, 'P-value'] = p

# Store predictions and results in csv files
prediction_file_name = 'tcga_dr_prediction(' + model_name + ').csv'
y_test_prediction.to_csv(results_path + prediction_file_name)
results_file_name = 'results(' + model_name + ').csv'
results.to_csv(results_path + results_file_name)

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

# Sensitive: complete response, partial response (1)
# Resistant: stable disease, progressive disease (0)

# Spend time learning multi-task methods

# Profile compute canada: find what is accessible to us, etc...
#       Takes time when requesting resources: in case need for next semester
#       Access GPUs: good for neural networks

# Right things down for report immediately: so that he can give us feedback

# 2 sided compare IC50 smaller bigger

############################### MEETING MARCH 20 ###############################
#
# Fix ElasticNetCV to save time
# Look at compute cancer canada: can we get GPUs?
# Check out Andrew Angee deep learning on youtube
# Start maybe moving toward deep learning: need to see what resources are available on CC to get a grant in time for next semester
# Can GPUs be accessible? Deadline to get is in november
# Do students get cloud assignment? Look into it
# May not need GPUs for small neural nets
# (1) Try 3-4 other single task models (other than elastic net)
    # Lasso
    # Support vector regression
    # Random forests
    # Support vector regression with RBF kernel (Gaussian kernel)
    # Note: can convert IC50 to binary to make it a clasification task
# Remember to fix normalization and t-test
# (2) Can start to implement a single task neural network architecture: look into different hyperparams (relu, etc...)
# (3) Send our report draft to Amin before deadline to get feedback
# What kind of data should we report?
#   Can make figures: 
#       Box-plot (MATPLOTLIB) y-axis = predicted IC50 of drug, x-axis = sensitive, resistant
#           red line = median, blue box contains standard deviation
#       Maybe include ElasticNet path graphs: may have to retrain after fitting?
