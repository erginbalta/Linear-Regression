# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 19:59:15 2018

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("linear_regression_dataset.csv", sep = ";")

plt.scatter (df.deneyim, df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show

from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()
x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)

plt.scatter(x,y)
plt.show()

y_head = linear_reg.predict(array)
plt.plot(array,y_head, color = "red")

linear_reg.predict(100)
