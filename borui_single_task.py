import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import ElasticNetCV
from sklearn.datasets import make_regression
from scipy import stats

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

def category_to_binary(y_test_single):
    y_test_binary = y_test_single.dropna()
    drug_name = list(y_test_binary.columns.values)[0]
    y_test_binary = y_test_binary.replace("Complete Response", 1)
    y_test_binary = y_test_binary.replace("Partial response", 1)
    y_test_binary = y_test_binary.replace("Stable Disease", 0)
    y_test_binary = y_test_binary.replace("Clinical Progressive Disease", 0)
    return y_test_binary

# Takes two numpy arrays as input: y_actual, y_predicted
# Returns two sample groups:
#   The first contains a list of predicted drug response values for which the patient's actual response was 0
#   The second contains a list of predicted drug response values for which the patient's actual response was 1
def get_t_test_groups(y_actual, y_predicted):
    
    # Ensure the sizes are the same
    #try:
    #    assert len(y_actual) == len(y_predicted)
    #except AssertionError:
    #    print("AssertionError: size of y_actual and y_predicted must be the same")

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


y_test_prediction = pd.DataFrame(index=y_test.index, columns=y_test.columns)

# Predict the response for each drug individually
for drug in y_train:

    # Keep only one drug column
    y_train_single = y_train[[drug]]
    y_test_single = y_test[[drug]]
    y_test_single_bi = category_to_binary(y_test[[drug]])

    # Drop cell line ids in y_train where drug response is NaN
    y_train_single = y_train_single.dropna()
    non_null_ids = y_train_single.index

    print("Training x columns...")
    # Drop cell line ids in x_train where drug response is NaN
    x_train_single = x_train[x_train.index.isin(non_null_ids)]

    # Create elastic net model with five-fold cross validation 
    regr = ElasticNetCV(cv=5, random_state=0)
    regr.fit(x_train_single.values, np.ravel(y_train_single.values))

    print("Predicting y columns...")
    # Predict the y_test drug response
    y_test_prediction_single = regr.predict(x_test)

    # Insert prediction into matrix of predictions
    # TO-DO set the column of y_test_prediction to y_test_prediction_single

    # Get the actual y_test binary values
   # y_test_actual_single = y_test_binary[[drug]]

    print("Getting responses ...")
    # Get sample groups for category 0 and category 1
    drug_responses_0, drug_responses_1 = get_t_test_groups(y_test_single_bi, y_test_prediction_single)
    # TO-DO: Call the t-test on drug_responses_0 and drug_response_1
    print(stats.ttest_ind(drug_responses_0,drug_responses_1))
    sys.exit(0)