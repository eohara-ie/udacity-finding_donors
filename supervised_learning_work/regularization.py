# TODO: Add import statements
#import numpy as np
#import pandas as pd
from sklearn.linear_model import Lasso
import csv

# Assign the data to predictor and outcome variables
# TODO: Load the data
#train_data = pd.read_csv('datareg.csv')
X = []
y = []
#print(train_data)
print('\n')
with open('datareg.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        X.append(row[:6])
        y.append(row[6])

print(X)
print('\n')
print(y)

# TODO: Create the linear regression model with lasso regularization.
lasso_reg = Lasso()

# TODO: Fit the model.
reg_model = lasso_reg.fit(X, y)

# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = reg_model.coef_
print(reg_coef)