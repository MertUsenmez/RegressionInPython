# -*- coding: utf-8 -*-
"""

@author: User
"""

# Evaluation Regression Model Performance with R-Square

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("random-forest-regression-dataset.csv", sep = ";", header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%%
from sklearn.ensemble import RandomForestRegressor

# We have divided 100 decision tree into 42 subdata.
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(x,y)

y_head = rf.predict(x)

#%% evaluate models

from sklearn.metrics import r2_score

# The closer to 1, the better.
print("r2 score = ", r2_score(y,y_head))