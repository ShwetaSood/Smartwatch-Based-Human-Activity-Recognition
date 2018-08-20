import os
import numpy as np
from numpy import loadtxt

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix

results = pd.read_csv("Kamal_mstdaccelerometerfinal.csv")
results = results.sample(frac=1)
tests = pd.read_csv("Shweta_mstdaccelerometerfinal.csv")
tests = tests.sample(frac=1)
data=results
test_data=tests


xlimit = 15
xtrain = data.iloc[:,[0,1,2,3,4,5,6,7,8,12,13,14]]
#[0,1,2,3,4,5,6,7,8,12,13,14]
ytrain = data.iloc[:, xlimit]

xtest = test_data.iloc[:, [0,1,2,3,4,5,6,7,8,12,13,14]]
ytest = test_data.iloc[:, xlimit]

model = XGBClassifier()
model.fit(xtrain, ytrain)
y_pred = model.predict(xtest)
predictions = [(int)(round(value)) for value in y_pred]

accuracy2 = np.mean(ytest == predictions)
print(confusion_matrix(ytest,predictions))
print(accuracy2)