# -*- coding: utf-8 -*-
"""

@author: User
"""

# Random Forest Regression is a Esemble Learning.
# Esemble learning : using multiple algorithms at the same time, the results are averaged and a model is created.
# It is use in recommended systems.
# It is use body part classification.
# It is use stock price prediction.

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

print(rf.predict(7.8))

x_ = np.arange(min(x), max(x), 0.01).reshape(-1,1)
y_head = rf.predict(x_)

#visualize
plt.scatter(x, y, color="red")
plt.plot(x_, y_head, color="green")
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show()