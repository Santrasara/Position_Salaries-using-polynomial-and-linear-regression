#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[22]:


data=pd.read_csv('Position_Salaries.csv')
data.head()


# In[23]:


X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values


# In[24]:


X


# In[25]:


y


# In[26]:


#Fitting Linear Regression to the data set
from sklearn.linear_model import LinearRegression
lin = LinearRegression()

lin.fit(X, y)


# In[27]:


#Visualizing Linear Regression Results
def linearReg():
    plt.scatter(X,y,color="cyan")
    plt.plot(X,lin.predict(X),color="magenta")
    plt.title("Linear Regression")
    plt.xlabel("Position")
    plt.ylabel("Salary")
    plt.show()
    return
linearReg()


# In[28]:


#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg =  PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)


# In[29]:


#Visualizing the Polynomial Regression results
def viz_polynomial():
    plt.scatter(X, y, color='cyan')
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='magenta')
    plt.title('Truth or Bluff (LinearRegression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    return
viz_polynomial()


# In[30]:


#Predicting a new result with Linear Regression
lin.predict([[5.5]])


# In[31]:


#Predicting a new result with Polynomial Regression
pol_reg.predict(poly_reg.fit_transform([[5.5]]))


# In[ ]:




