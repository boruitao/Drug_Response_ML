import numpy as np
import pandas as pd
import time

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import MultiTaskLasso, MultiTaskLassoCV
from sklearn.datasets import make_regression
from sklearn.impute import SimpleImputer
from scipy import stats

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
    for gene in x_train: # Same genes in both x_train and x_test, so loop once
        x_train[gene] = ss.fit_transform(x_train[[gene]].values) # Figure out the normalization parameters on the training set
        x_test[gene] = ss.transform(x_test[[gene]].values) # Use the same normalization parameters used on the training set  
    x_train.T.to_csv(data_path + 'gdsc_expr_postCB(normalized)1.csv')
    x_test.T.to_csv(data_path + 'tcga_expr_postCB(normalized)1.csv')

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

start_time = time.time()

# Path to datasets
data_path = '../Data/'

# Path to results folder
results_path = '../Results/'

# Name of model being used
model_name = 'MultitaskLassoCV-cv=3,n_alphas=10'
# model_name = 'MultitaskLasso'

# Load training and test set
x_train = pd.read_csv(data_path + 'gdsc_expr_postCB(normalized).csv', index_col=0, header=None).T.set_index('cell line id').apply(pd.to_numeric)
y_train = pd.read_csv(data_path + 'gdsc_dr_lnIC50.csv', index_col=0, header=None).T.set_index('cell line id').apply(pd.to_numeric)
x_test = pd.read_csv(data_path + 'tcga_expr_postCB(normalized).csv', index_col=0, header=None).T.set_index('patient id').apply(pd.to_numeric)
y_test = pd.read_csv(data_path + 'tcga_dr.csv', index_col=0, header=None).T.set_index('patient id')
y_test_binary = category_to_binary(y_test)

#normalize(x_train, x_test)

# Impute missing values in y_train
imp = SimpleImputer(missing_values=np.nan, strategy='mean') # If a drug's DR is NaN, set it to the mean of the cell lines' DR for that drug
y_train = pd.DataFrame(data=imp.fit_transform(y_train), index=y_train.index, columns=y_train.columns)

# Matrix to store drug statistics, including t-statistic and p-value for each drug
results = y_test.describe().T.join(pd.DataFrame(index=y_test.columns, columns=['T-statistic', 'P-value']))
results = results.drop(["count", "unique", "top", "freq"], axis=1)

# Drop cell line id in x_train and y_train if one of the drug responses is NaN
# y_train = y_train.dropna() # Drop rows of y_train
# non_null_ids = y_train.index # Get cell line ids that don't have null drug responses
# x_train = x_train[x_train.index.isin(non_null_ids)]

# Create multitask lasso model
print("Fitting " + model_name + "...")
# regr = MultiTaskLasso(alpha=0.5).fit(x_train, y_train)
regr = MultiTaskLassoCV(cv=3, n_alphas=10).fit(x_train, y_train)

# Predict y_test
print("Predicting y test...")
y_test_prediction = pd.DataFrame(data=regr.predict(x_test), index=y_test.index, columns=y_test.columns)

# For each drug, execute a t-test and store the results
for drug in y_test_binary.columns:

    # Get the drug response vector for a single drug
    y_test_prediction_single = y_test_prediction[drug] # assign column headers to y_test_prediction
    y_test_actual_single = y_test_binary[drug]

    # Get sample groups for category 0 and category 1
    drug_responses_0, drug_responses_1 = get_t_test_groups(y_test_actual_single.values, y_test_prediction_single)

    # Perform T-test
    print("Performing t-test for drug: " + str(drug))
    drug_responses_0, drug_responses_1 = get_t_test_groups(y_test_actual_single.values, y_test_prediction_single)
    t, p = one_tailed_t_test(drug_responses_0, drug_responses_1)
    results.loc[drug, 'T-statistic'] = t
    results.loc[drug, 'P-value'] = p

# Store predictions and results in csv files
# prediction_file_name = 'tcga_dr_prediction(' + model_name + ').csv'
# y_test_prediction.to_csv(results_path + prediction_file_name)
results_file_name = 'results(' + model_name + ').csv'
results.to_csv(results_path + results_file_name)

# Print the total running time
end_time = time.time()
total_time = end_time - start_time
print("Total running time was: " + str(total_time) + " seconds")