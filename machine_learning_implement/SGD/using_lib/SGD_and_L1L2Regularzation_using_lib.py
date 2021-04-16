#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


from sklearn.datasets import load_boston
boston = load_boston()


# In[10]:


type(boston)


# In[8]:


boston.feature_names


# In[9]:


df = pd.DataFrame(boston.data,columns=boston.feature_names)
df.head()


# In[13]:


x = df.values
y =  boston.target
y[:10]


# In[14]:


x[:10]


# In[15]:


from sklearn.linear_model import LinearRegression #Normal equation
lr_ne = LinearRegression(fit_intercept=True)


# In[16]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)


# In[17]:


lr_ne.fit(X_train,y_train)


# In[18]:


boston.feature_names


# In[19]:


lr_ne.intercept_, lr_ne.coef_


# In[20]:


y_hat = lr_ne.predict(X_test)
y_true = y_test


# In[21]:


rmse = np.sqrt(sum((y_hat - y_true)**2)/len(y_true))
rmse


# In[24]:


import sklearn
mse = sklearn.metrics.mean_squared_error(y_hat,y_true)
mse


# In[26]:


plt.scatter(y_true,y_hat,s=10)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")


# In[27]:


##SGF


# In[28]:


from sklearn.linear_model import SGDRegressor
lr_SGD = SGDRegressor()


# In[29]:


from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
std_scaler.fit(X)
X_scaled = std_scaler.transform(X)


# In[36]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.33,random_state=42)


# In[37]:


lr_SGD.fit(X_train,y_train)


# In[38]:


y_hat = lr_SGD.predict(X_test)
y_true = y_test


# In[39]:


mse = sklearn.metrics.mean_squared_error(y_hat,y_true)
mse


# In[41]:


plt.scatter(y_true,y_hat,s=10)


# In[42]:


#Linear Regression with Ridge & Lasso regression


# In[43]:


from sklearn.linear_model import Lasso,Ridge


# In[44]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)


# In[45]:


rdige = Ridge(fit_intercept=True,alpha=0.5) #l2 reggularzation


# In[46]:


lasso = Lasso(fit_intercept=True,alpha=0.5)


# In[47]:


rdige.fit(X_train,y_train)


# In[49]:


y_hat = rdige.predict(X_test)
y_true = y_test

mse = sklearn.metrics.mean_squared_error(y_hat,y_true)
mse


# In[50]:


plt.scatter(y_true,y_hat,s=10)


# In[51]:


from sklearn.model_selection import KFold


# In[52]:


print('ridge regression')
alpha = np.linspace(.01,20,50)
t_rmse = np.array([])
cv_rmse = np.array([])

for a in alpha:
    ridge = Ridge(fit_intercept=True,alpha=a)
    
    ridge.fit(X_train,y_train)
    p = ridge.predict(X_test)
    err = p-y_test
    total_error = np.dot(err,err)
    rmse_train = np.sqrt(total_error/len(p))
    
    kf = KFold(10)
    xval_err = 0
    for train,test in kf.split(X):
        ridge.fit(x[train],y[train])
        p = ridge.predict(x[test])
        err = p-y[test]
        xval_err += np.dot(err,err)
    rmse_10cv = np.sqrt(xval_err/len(x))
    
    t_rmse = np.append(t_rmse,[rmse_train])
    cv_rmse = np.append(cv_rmse,[rmse_10cv])
    print('{:.3f}\t {:.4f}\t\t{:.4f}'.format(a,rmse_train,rmse_10cv))


# In[53]:


plt.plot(alpha,t_rmse,label='RMSE_Train')
plt.plot(alpha,cv_rmse,label='RMSE_XVal')
plt.legend(('RMSE-Train','RMSE_Xval'))
plt.ylabel('RMSE')
plt.xlabel('Alpha')
plt.show()

