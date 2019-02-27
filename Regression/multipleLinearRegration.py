# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 21:16:08 2018

@author: User
"""
 # multiple linear regression equation = y = bo + b1*x1 + b2*x2

 import pandas as pd
 import numpy as np
 from sklearn.linear_model import LinearRegression
 
 df = pd.read_csv("multiple-linear-regression-dataset.csv", sep=";")
 
x = df.iloc[:,[0.2]].values
y = df.maas.values.reshape(-1,1)

#%%

multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)

print("b0 = ", multiple_linear_regression.intercept_)
print("b1,b2 = ", multiple_linear_regression.coef_)

#predict
multiple_linear_regression.predict(np.array([[10,35],[5,35]]))