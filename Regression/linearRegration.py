import pandas as pd

df = pd.DataFrame(linearregressiondatasetcsv, columns=["deneyim","maas"])

import matplotlib.pyplot as plt
 
plt.scatter(df.deneyim, df.maas)

#%%

df = pd.read_csv("linear-regression-dataset.csv", sep = ";")

plt.scatter(df.deneyim,df.maas)
plt.xlabel("Experiance")
plt.ylabel("Salary")
plt.show()

#%%  Linear Regression

# (y = b0 + b1*x) creates a graph of it creates the equation

#sklearn library
from sklearn.linear_model import LinearRegression

#linear regression model
linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

#%%

b0 = linear_reg.predict(0)
print("b0=",b0)

#This is the point where the y axis cuts so intercept point
b0_ = linear_reg.intercept_
print("b0_=",b0_)

#slope
b1 = linear_reg.coef_
print("b1=",b1)

# (maas(in English salary) = 1138 + 1663*deneyim(in English experience)) we have this equation now

# şimdi yeni bir prediction yapalım
# Now, lets make new prediction.
# We make prediction of salary of thirteen years experience.
salary_new = b1 + b0 * 13
print("New salary : ", salary_new)

# Or we can make prediction with using sklearn library (for eleven years experience).
print(linear_reg.predict(11))


#viasualize line
import numpy as np

array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)

plt.scatter(x,y)
plt.show()

#salary
y_head = linear_reg.predict(array)
plt.plot(array, y_head, color="red")








