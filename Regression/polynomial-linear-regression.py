# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 18:19:29 2018

@author: User
"""

 import pandas as pd
 import matplotlib.pyplot as plt
 
 df = pd.read_csv("polynomial-regression.csv", sep = ";")
 
 y = df.araba_max_hiz.values.reshape(-1, 1)
 x = df.araba_fiyat.values.reshape(-1, 1)
 
 plt.scatter(x, y)
 plt.ylabel("car_max_speed")
 plt.xlabel("car_cost")
 plt.show()
 
# Reninder
# linear regression -> y = b0 + b1*x
# multiple linear regression -> y = b0 + b1*x1 + b2*x2
 
 #%% Polynomial Regression
 
 from sklearn.linear_model import LinearRegression
 
 lr = LinearRegression()
 lr.fit(x,y)
 
 y_head = lr.predict(x)
 
 plt.plot(x, y_head, color="red", label="linear")
 plt.show()
 
 
 lr.predict(10000)
 print("Speed for cost of car is 10 billion dolar = ", lr.predict(10000))
 # we will see a meaningless result.  
 # There's no such thing as saying that we're going to have a car with an 871-km-hour speed.
 # So, linear regression is not enough for this problem.
 # Therefore, we will use polynomial regression.
 # Our new equation is ->  y=b0+ b1*x1 + b2*x2^2 + b3*x3^3 + .... + bn*xn^n
 # b2*x2^2  => parabolic
 # This mean curvifying this line, increasing the slope
 # After a period of time the speed is starting to remain constant because of the polynomial.
 
 
 #%% Polynomial Regression
 # Firstly, we need to open x^2 coloum.
 
 from sklearn.preprocessing import PolynomialFeatures
 
 polynomial_regression = PolynomialFeatures(degree = 4) # You need to change this degree according to your purpose.
 
 x_polynomial = polynomial_regression.fit_transform(x)
 
 #%% fit
 
 linearRegression2 = LinearRegression()
 linearRegression2.fit(x_polynomial, y)
 
 #%% visualization
 
 y_head2 = linearRegression2.predict(x_polynomial)
 
 plt.plot(x, y_head2, color="green", label="polynomial")
 plt.legend()
 plt.show()
 
 
 
 
 
 
 