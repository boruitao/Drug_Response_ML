import numpy as np
import pandas as pd
import time
import math
import itertools
import matplotlib.pyplot as plt

# Ignore sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.datasets import make_regression
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

def grouper(n, iterable):
    it = iter(iterable)
    while True:
       chunk = tuple(itertools.islice(it, n))
       if not chunk:
           return
       yield chunk

# Prints an array with n entries per line
def print_array_n_entries_per_line(array, n):
    for chunk in grouper(n, array): 
        print(" ".join(str(x) for x in chunk))

start_time = time.time()

# Path to datasets
data_path = '../Data/'

# Path to results folder
results_path = '../Results/'

# Name of model being used
model_name = 'LassoCV'

# Load training and test set
x_train = pd.read_csv(data_path + 'gdsc_expr_postCB(normalized).csv', index_col=0, header=None, low_memory=False).T.set_index('cell line id').apply(pd.to_numeric)#.iloc[:,0:100]
y_train = pd.read_csv(data_path + 'gdsc_dr_lnIC50.csv', index_col=0, header=None, low_memory=False).T.set_index('cell line id').apply(pd.to_numeric)
x_test = pd.read_csv(data_path + 'tcga_expr_postCB(normalized).csv', index_col=0, header=None, low_memory=False).T.set_index('patient id').apply(pd.to_numeric)#.iloc[:,0:100]
y_test = pd.read_csv(data_path + 'tcga_dr.csv', index_col=0, header=None, low_memory=False).T.set_index('patient id')
y_test_binary = category_to_binary(y_test)

# Drop the drugs with low sample sizes (to speed of running time)
columns = ['dabrafenib','erlotinib','gefitinib','imatinib','lapatinib','methotrexate','sunitinib','trametinib','veliparib','vinblastine']
y_train = y_train.drop(columns, 1)
y_test = y_test.drop(columns, 1)

# Matrix to store y_test predictions
y_test_prediction = pd.DataFrame(index=y_test.index, columns=y_test.columns)

# Matrix to store drug statistics, including t-statistic and p-value for each drug
results = y_test.describe().T.join(pd.DataFrame(index=y_test.columns, columns=['T-statistic', 'P-value']))
results = results.drop(["count", "unique", "top", "freq"], axis=1)

# Predict the response for each drug individually
drug_counter = 1
for drug in y_train:

    # Keep only one drug column
    y_train_single = y_train[[drug]]

    # Drop rows in y_train where drug response is NaN, as well as corresponding x_train rows
    y_train_single = y_train_single.dropna()
    non_null_ids = y_train_single.index
    x_train_single = x_train[x_train.index.isin(non_null_ids)]

    # Create and fit model
    print("========================================================================================================")
    print("\nFitting " + model_name + " for drug: " + drug)
    
    regr = LassoCV(cv=5, random_state=0)
    regr.fit(x_train_single.values, np.ravel(y_train_single.values))

    # Display results
    EPSILON = 1e-4
    m_log_alphas = -np.log10(regr.alphas_ + EPSILON)

    plt.figure()
    ymin, ymax = 0, 8
    plt.plot(m_log_alphas, regr.mse_path_.mean(axis=-1), 'k',
            label='Mean MSE across the folds', linewidth=2)
    plt.axvline(-np.log10(regr.alpha_ + EPSILON), linestyle='--', color='k',
                label='alpha chosen by CV') # Vertical line
    plt.legend()

    mse_path = np.array(regr.mse_path_).mean(axis=1) # Get MSE path with mean CV scores
    print("\n[MSE path]:")
    print(mse_path)
    print("\n[Lowest MSE]:")
    test_MSE = round(mse_path.min(), 5)
    print(test_MSE)

    title = str(drug) + ' - ' + model_name + ' coordinate descent - MSE = ' + str(test_MSE)
    plt.xlabel('-log(alpha)')
    plt.ylabel('Mean squared error')
    plt.title(title)
    plt.axis('tight')
    plt.ylim(ymin, ymax)
    #plt.show()
    plt.savefig(results_path + title + '.png', dpi=1200, format='png', bbox_inches='tight')

    # Predict y_test drug response, and insert into prediction matrix
    print("\nPredicting TCGA drug response...")
    y_test_prediction_single =  regr.predict(x_test)
    y_test_prediction[drug] = y_test_prediction_single
    
    # Perform T-test
    print("\nPerforming t-test...")
    y_test_actual_single = y_test_binary[[drug]]
    drug_responses_0, drug_responses_1 = get_t_test_groups(y_test_actual_single.values, y_test_prediction_single)
    t, p = one_tailed_t_test(drug_responses_0, drug_responses_1)
    results.loc[drug, 'T-statistic'] = t
    results.loc[drug, 'P-value'] = p
    print("T-statistic: " + str(t))
    print("P-value: " + str(p))

    current_time = time.time()
    print("\nCurrent time: " + str(current_time - start_time) + "\n")

    drug_counter += 1

# Store results in csv file
results_file_name = 'results(' + model_name + ' ' + str(time.time()) + ').csv'
results.to_csv(results_path + results_file_name)

print("========================================================================================================")

# Print the total running time
end_time = time.time()
total_time = end_time - start_time
print("\nTotal running time was: " + str(total_time) + " seconds")