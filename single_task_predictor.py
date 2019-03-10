import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

################# SINGLE TASK DRUG RESPONSE PREDICTOR ################# 

drug_col = 0 # Column of the drug to predict (0 = Bicalutamide)

# Load training and test set
X_train = np.load('gdsc_expr.npy')
y_train = np.load('gdsc_dr.npy')[:,drug_col]
X_test = np.load('tcga_expr.npy')

# Remove rows in X_train and Y_train where the Y_train value is NaN (no drug response data for that cell line)
mask = np.logical_not(np.ma.masked_invalid(y_train).mask)
y_train = y_train[mask]
X_train = X_train[mask,:]

# #############################################################################
# LINEAR REGRESSION
# Source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py

regr = linear_model.LinearRegression() # Create linear regression object
regr.fit(X_train, y_train) # Train the model
y_pred_regr = regr.predict(X_test) # Make predictions for the test set

print('Coefficients: \n', regr.coef_)
print('Linear Regression Predictions: \n', y_pred_regr)
#print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
#print('Variance score: %.2f' % r2_score(y_test, y_pred))

# Plot outputs
#plt.scatter(X_test, y_test,  color='black')
#plt.plot(X_test, y_pred, color='blue', linewidth=3)
#plt.xticks(())
#plt.yticks(())
#plt.show()

# #############################################################################
# LASSO
# Source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#sphx-glr-auto-examples-linear-model-plot-lasso-and-elasticnet-py

alpha = 0.1
lasso = Lasso(alpha=alpha)

y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
print('Lasso Predictions: \n', y_pred_lasso)
#r2_score_lasso = r2_score(y_test, y_pred_lasso)
#print(lasso)
#print("r^2 on test data : %f" % r2_score_lasso)

# #############################################################################
# ELASTIC NET
# Source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#sphx-glr-auto-examples-linear-model-plot-lasso-and-elasticnet-py

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
print('Lasso Predictions: \n', y_pred_enet)
#r2_score_enet = r2_score(y_test, y_pred_enet)
#print(enet)
#print(y_pred_enet)
#print("r^2 on test data : %f" % r2_score_enet)

# #############################################################################
# Plot results
#plt.plot(enet.coef_, color='lightgreen', linewidth=2, label='Elastic net coefficients')
#plt.plot(lasso.coef_, color='gold', linewidth=2, label='Lasso coefficients')
#plt.legend(loc='best')
#plt.title("Lasso R^2: %f, Elastic Net R^2: %f" % (r2_score_lasso, r2_score_enet))
#plt.show()

