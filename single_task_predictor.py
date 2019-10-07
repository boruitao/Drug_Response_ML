import numpy as np
import pandas as pd
import time
import math

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.neural_network import MLPRegressor
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
model_name = 'MLPRegressor - random_state=0, activation=logistic, hidden_layer_sizes=(25,)'

# Load training and test set
x_train = pd.read_csv(data_path + 'gdsc_expr_postCB(normalized).csv', index_col=0, header=None, low_memory=False).T.set_index('cell line id').apply(pd.to_numeric)
y_train = pd.read_csv(data_path + 'gdsc_dr_lnIC50.csv', index_col=0, header=None, low_memory=False).T.set_index('cell line id').apply(pd.to_numeric)
x_test = pd.read_csv(data_path + 'tcga_expr_postCB(normalized).csv', index_col=0, header=None, low_memory=False).T.set_index('patient id').apply(pd.to_numeric)
y_test = pd.read_csv(data_path + 'tcga_dr.csv', index_col=0, header=None, low_memory=False).T.set_index('patient id')
y_test_binary = category_to_binary(y_test)

# Drop the drugs with low sample sizes (to speed of running time)
columns = ['dabrafenib','erlotinib','gefitinib','imatinib','lapatinib','methotrexate','sunitinib','trametinib','veliparib','vinblastine']
y_train = y_train.drop(columns, 1)
y_test = y_test.drop(columns, 1)

# Split into training and validation set
training_split = 0.8
n_samples_training = math.ceil(training_split * x_train.shape[0])
n_samples_val = x_train.shape[0] - n_samples_training

x_train_ = x_train.head(n_samples_training)
y_train_ = y_train.head(n_samples_training)
x_val = x_train.tail(n_samples_val)
y_val = y_train.tail(n_samples_val)

# Matrix to store y_test predictions
y_test_prediction = pd.DataFrame(index=y_test.index, columns=y_test.columns)

# Matrix to store drug statistics, including t-statistic and p-value for each drug
results = y_test.describe().T.join(pd.DataFrame(index=y_test.columns, columns=['T-statistic', 'P-value']))
results = results.drop(["count", "unique", "top", "freq"], axis=1)

# Predict the response for each drug individually
for drug in y_train_:

    # Keep only one drug column
    y_train_single = y_train_[[drug]]

    # Drop rows in y_train where drug response is NaN, as well as corresponding x_train rows
    y_train_single = y_train_single.dropna()
    non_null_ids = y_train_single.index
    x_train_single = x_train_[x_train_.index.isin(non_null_ids)]

    # Drop rows in y_val where drug response is NaN, as well as corresponding x_val rows
    y_val_single = y_val[[drug]]
    y_val_single = y_val_single.dropna()
    non_null_ids_ = y_val_single.index
    x_val_single = x_val[x_val.index.isin(non_null_ids_)]

    # Create elastic net model with five-fold cross validation
    print("Fitting " + model_name + " for drug: " + drug)
    regr = MLPRegressor(random_state=0, activation='logistic', hidden_layer_sizes=(25,))
    regr.fit(x_train_single.values, np.ravel(y_train_single.values))

     # Accuracy
    print("Training accuracy: " + str(regr.score(x_train_single.values, np.ravel(y_train_single.values))))
    print("Validation accuracy: " + str(regr.score(x_val_single.values, np.ravel(y_val_single.values))))

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
    print("T-statistic: " + str(t))
    print("P-value: " + str(p))

    current_time = time.time()
    print("Current time: " + str(current_time - start_time) + "\n")

# Store predictions and results in csv files
results_file_name = 'results(' + model_name + ').csv'
results.to_csv(results_path + results_file_name)

# Print the total running time
end_time = time.time()
total_time = end_time - start_time
print("Total running time was: " + str(total_time) + " seconds")