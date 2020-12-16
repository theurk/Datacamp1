#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


#1


# In[9]:


df = pd.read_csv('USA_Housing.csv')
df


# In[10]:


#2


# In[11]:


df.head(10)


# In[12]:


df.tail(10)


# In[13]:


df.sample(10)


# In[14]:


#3 และ #4  เช็คแล้วไม่มี missing value


# In[15]:


df.info()


# In[16]:


df.describe()


# In[17]:


#5


# In[18]:


sns.pairplot(df)


# In[19]:


#6


# In[25]:


fig = plt.figure(figsize=(12,8))
sns.distplot(df['Price'])


# In[26]:


sns.distplot(df['Area Population'])


# In[27]:


sns.distplot(df['Avg. Area Number of Bedrooms'])


# In[28]:


sns.distplot(df['Avg. Area Number of Rooms'])


# In[29]:


sns.distplot(df['Avg. Area House Age'])


# In[30]:


sns.distplot(df['Avg. Area Income'])


# In[31]:


#7


# In[32]:


sns.heatmap(df.corr())


# In[33]:


#8


# In[35]:


fig = plt.figure(figsize=(12,8))
plt.title('Best correlation')
plt.xlabel('Avg. Area Income')
plt.ylabel('Price')
plt.scatter(df['Price'], df['Avg. Area Income'])


# In[36]:


#9


# In[37]:


fig = plt.figure(figsize=(12,8))
plt.title('Worst correlation')
plt.xlabel('Avg. Area Income')
plt.ylabel('Price')
plt.scatter(df['Avg. Area Income'], df['Avg. Area Number of Rooms'])


# In[38]:


#10


# In[40]:


fig = plt.figure(figsize=(12,8))
plt.hist(df['Price'],bins=100)
plt.show()


# In[ ]:


#11


# In[44]:


fig = plt.figure(figsize=(12,8))
sns.boxplot(df['Price'], orient='v')


# In[ ]:


#12


# In[48]:


print('Mean', df['Price'].mean())
print('Median', df['Price'].median())
print('Mode', df['Price'].mode()[0])


# In[49]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[51]:


X = df['Avg. Area Income']
y = df['Price']


# In[52]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[53]:


X_train


# In[54]:


type(X_train)


# In[55]:


X_train = np.array(X_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)


# In[56]:


X_train.shape


# In[57]:


X_train


# In[58]:


#13


# In[59]:


lm = LinearRegression()
lm.fit(X_train, y_train)


# In[61]:


# y = ax + b

# b = intercept
# a = coef = slope


# In[62]:


print(lm.intercept_)
print(lm.coef_)


# In[63]:


predicted = lm.predict(X_test)
predicted


# In[ ]:


#14


# In[68]:


print('MAE: ', metrics.mean_absolute_error(y_test, predicted))
print('MSE: ', metrics.mean_squared_error(y_test, predicted))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, predicted)))


# In[69]:


#15


# In[72]:


fig = plt.figure(figsize=(12,8))
sns.distplot((y_test-predicted), bins=50);


# In[75]:


#16


# In[81]:


fig = plt.figure(figsize=(12,8))
plt.scatter(X_test, y_test, color='green', label= 'data')
plt.plot(X_test, predicted, color='red', label= 'Predicted Regression Line')
plt.xlabel('Avg. Area Income')
plt.ylabel('Price')
plt.legend()


# In[82]:


#17


# In[83]:


df


# In[85]:


X2 = df[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']]
y2 = df['Price']


# In[86]:


X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=100)


# In[88]:


lm = LinearRegression()
lm.fit(X2_train, y2_train)


# In[89]:


print(lm.intercept_)
print(lm.coef_)


# In[92]:


coeff_df = pd.DataFrame(lm.coef_, X2.columns,columns= ['Coefficient'])
coeff_df


# In[93]:


#18


# In[94]:


predicted2 = lm.predict(X2_test)
predicted2


# In[95]:


print('MAE: ', metrics.mean_absolute_error(y2_test, predicted2))
print('MSE: ', metrics.mean_squared_error(y2_test, predicted2))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y2_test, predicted2)))


# In[96]:


#19


# In[97]:


fig = plt.figure(figsize=(12,8))
sns.distplot((y2_test-predicted2), bins=50);


# In[98]:


#20


# In[99]:


RMSE_Multi = np.sqrt(metrics.mean_squared_error(y2_test, predicted2))
RMSE_Multi


# In[101]:


RMSE_Single = np.sqrt(metrics.mean_squared_error(y_test, predicted))
RMSE_Single


# In[102]:


RMSE_Multi - RMSE_Single


# In[ ]:


# RMSE_Multi มีค่าน้อยกว่า RMSE_Single อยู่ 163103.3747565427

# สรุปว่า RMSE ของ Multiple Regression แม่นยำกว่า Single Regression

