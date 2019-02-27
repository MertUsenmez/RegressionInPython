# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 22:34:08 2018

@author: Mert
"""

# Purpose of machine teaching mean square error(MSE) to make zero.
# MSE = square root of standart deviation. Other name is variance. Refers to the distribution of the average.
# MSE =(1/n)*((y1-y1')^2 + (y2-y2')^2 + (y3-y3')^2 + ... + (yn-yn')^2

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("decision-tree-regression-dataset.csv", sep = ";", header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%% decision tree regression

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()  #Random state = 0
tree_reg.fit(x,y)

tree_reg.predict(5.5)
x_ = np.arange(min(x),max(x), 0.01).reshape(-1,1)  # Create values from 1 to 10
y_head = tree_reg.predict(x_)

#%% visualize

plt.scatter(x,y,color="red")
plt.plot(x_, y_head, color="green")
plt.xlabel("Tribun level")
plt.ylabel("cost")
plt.show()

 # The resulting graph shows the sudden declines and outputs from a terminal leaf to another terminal leaf.
 # Decision mechanism
 # We used 1 dimensional dataset and visualized in 2 dimensional.