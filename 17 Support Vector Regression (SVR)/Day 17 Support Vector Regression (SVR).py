#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#1


# In[3]:


df = pd.read_csv('House price prediction.csv')
df


# In[4]:


#2


# In[5]:


df.head(10)


# In[6]:


df.tail(10)


# In[7]:


df.sample(10)


# In[10]:


#3 และ #4  เช็คแล้วไม่มี missing value


# In[11]:


df.info()


# In[12]:


df.describe()


# In[13]:


#5


# In[14]:


sns.pairplot(df)


# In[15]:


#6


# In[21]:


fig = plt.figure(figsize=(12,8))
sns.distplot(df['price'])


# In[22]:


sns.distplot(df['bedrooms'])


# In[23]:


sns.distplot(df['bathrooms'])


# In[24]:


sns.distplot(df['sqft_living'])


# In[25]:


sns.distplot(df['sqft_lot'])


# In[26]:


sns.distplot(df['floors'])


# In[27]:


sns.distplot(df['waterfront'])


# In[28]:


sns.distplot(df['view'])


# In[29]:


sns.distplot(df['condition'])


# In[30]:


sns.distplot(df['sqft_above'])


# In[31]:


sns.distplot(df['sqft_basement'])


# In[32]:


sns.distplot(df['yr_built'])


# In[33]:


sns.distplot(df['yr_renovated'])


# In[34]:


#7


# In[35]:


sns.heatmap(df.corr())


# In[36]:


#8


# In[39]:


fig = plt.figure(figsize=(12,8))
plt.title('Best correlation')
plt.xlabel('sqft_living')
plt.ylabel('price')
plt.scatter(df['price'], df['sqft_living'])


# In[40]:


#9


# In[41]:


fig = plt.figure(figsize=(12,8))
plt.title('Worst correlation')
plt.xlabel('floors')
plt.ylabel('sqft_basement')
plt.scatter(df['sqft_basement'], df['floors'])


# In[42]:


#10


# In[46]:


fig = plt.figure(figsize=(12,8))
plt.hist(df['price'],bins=100)
plt.show()


# In[47]:


#11


# In[48]:


fig = plt.figure(figsize=(12,8))
sns.boxplot(df['price'], orient='v')


# In[49]:


#12


# In[447]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[448]:


X = df['sqft_living']
y = df['price']


# In[449]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[450]:


X_train


# In[451]:


y_train


# In[452]:


#13


# In[453]:


X_train = np.array(X_train).reshape(-1,1)
y_train = np.array(y_train).reshape(-1,1)


# In[454]:


X_train


# In[455]:


y_train


# In[456]:


from sklearn.preprocessing import StandardScaler
sc_X_train = StandardScaler()
sc_y_train = StandardScaler()

X_train = sc_X_train.fit_transform(X_train)
y_train = sc_y_train.fit_transform(y_train)


# In[457]:


X_train


# In[458]:


y_train


# In[459]:


#14


# In[460]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'linear')
regressor.fit(X_train, y_train)


# In[461]:


X_test = np.array(X_test).reshape(-1,1)


# In[462]:


sc_X_test = StandardScaler()

X_test = sc_X_test.fit_transform(X_test)


# In[463]:


result = regressor.predict(X_test)
print(result)


# In[509]:


sc_y_train.inverse_transform(result)


# In[510]:


predicted = sc_y_train.inverse_transform(result)
predicted 


# In[511]:


fig = plt.figure(figsize=(12,8))
plt.scatter(sc_X_test.inverse_transform(X_test), y_test, color='green', label= 'data')
plt.plot(sc_X_test.inverse_transform(X_test), predicted , color='red')
plt.xlabel('sqft_living')
plt.ylabel('price')
plt.legend()


# In[496]:


# rbf


# In[512]:


X = df['sqft_living']
y = df['price']


# In[513]:


X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=100)


# In[514]:


X_train2


# In[515]:


X_train2 = np.array(X_train2).reshape(-1,1)
y_train2 = np.array(y_train2).reshape(-1,1)


# In[516]:


X_train2


# In[517]:


sc_X_train2 = StandardScaler()
sc_y_train2 = StandardScaler()

X_train2 = sc_X_train2.fit_transform(X_train2)
y_train2 = sc_y_train2.fit_transform(y_train2)


# In[518]:


X_train2


# In[519]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train2, y_train2)


# In[520]:


X_test2 = np.array(X_test2).reshape(-1,1)


# In[521]:


sc_X_test2 = StandardScaler()

X_test2 = sc_X_test2.fit_transform(X_test2)


# In[522]:


result2 = regressor.predict(X_test2)
print(result2)


# In[523]:


predicted2 = sc_y_train2.inverse_transform(result2)
predicted2 


# In[524]:


fig = plt.figure(figsize=(12,8))
plt.scatter(sc_X_test2.inverse_transform(X_test2), y_test2, color='green', label= 'data')
plt.plot(sc_X_test2.inverse_transform(X_test2), predicted2 , color='red')
plt.xlabel('sqft_living')
plt.ylabel('price')
plt.legend()


# In[525]:


print('MAE: ', metrics.mean_absolute_error(y_test, predicted))
print('MSE: ', metrics.mean_squared_error(y_test, predicted))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, predicted)))


# In[526]:


print('MAE: ', metrics.mean_absolute_error(y_test2, predicted2))
print('MSE: ', metrics.mean_squared_error(y_test2, predicted2))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test2, predicted2)))


# In[527]:


#15


# In[528]:


df


# In[529]:


X3 = df[['bathrooms','sqft_living','sqft_above']]
y3 = df['price']


# In[530]:


X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=100)


# In[531]:


X3_train


# In[532]:


y3_train = np.array(y3_train).reshape(-1,1)


# In[533]:


y3_train


# In[534]:


sc_X3_train = StandardScaler()
sc_y3_train = StandardScaler()

X3_train = sc_X3_train.fit_transform(X3_train)
y3_train = sc_y3_train.fit_transform(y3_train)


# In[535]:


X3_train


# In[536]:


y3_train


# In[537]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'linear')
regressor.fit(X3_train, y3_train)


# In[538]:


sc_X3_test = StandardScaler()

X3_test = sc_X3_test.fit_transform(X3_test)


# In[539]:


result3 = regressor.predict(X3_test)
print(result3)


# In[541]:


predicted3 = sc_y3_train.inverse_transform(result3)
predicted3 


# In[542]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X3_train, y3_train)


# In[543]:


result4 = regressor.predict(X3_test)
print(result4)


# In[544]:


predicted4 = sc_y3_train.inverse_transform(result4)
predicted4 


# In[547]:


#16   แบบ linear


# In[548]:


print('MAE: ', metrics.mean_absolute_error(y3_test, predicted3))
print('MSE: ', metrics.mean_squared_error(y3_test, predicted3))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y3_test, predicted3)))


# In[549]:


#16   แบบ rbf


# In[550]:


print('MAE: ', metrics.mean_absolute_error(y3_test, predicted4))
print('MSE: ', metrics.mean_squared_error(y3_test, predicted4))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y3_test, predicted4)))


# In[551]:


#17 Single linear


# In[552]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[553]:


X = df['sqft_living']
y = df['price']


# In[554]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[555]:


X_train


# In[556]:


X_train = np.array(X_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)


# In[557]:


X_train.shape


# In[558]:


X_train


# In[559]:


lm = LinearRegression()
lm.fit(X_train, y_train)


# In[560]:


print(lm.intercept_)
print(lm.coef_)


# In[561]:


# y = ax + b

# b = intercept
# a = coef = slope


# In[562]:


predicted = lm.predict(X_test)
predicted


# In[563]:


print('MAE: ', metrics.mean_absolute_error(y_test, predicted))
print('MSE: ', metrics.mean_squared_error(y_test, predicted))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, predicted)))


# In[564]:


#17  แบบ multiple


# In[565]:


X3 = df[['bathrooms','sqft_living','sqft_above']]
y3 = df['price']


# In[566]:


X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=100)


# In[567]:


X3_train


# In[568]:


lm = LinearRegression()
lm.fit(X3_train, y3_train)


# In[569]:


print(lm.intercept_)
print(lm.coef_)


# In[571]:


coeff_df = pd.DataFrame(lm.coef_, X3.columns,columns= ['Coefficient'])
coeff_df


# In[572]:


predicted2 = lm.predict(X3_test)
predicted2


# In[574]:


#18 แบบ simple regression


# In[575]:


print('MAE: ', metrics.mean_absolute_error(y_test, predicted))
print('MSE: ', metrics.mean_squared_error(y_test, predicted))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, predicted)))


# In[576]:


#18 แบบ multiple regression


# In[577]:


print('MAE: ', metrics.mean_absolute_error(y_test, predicted2))
print('MSE: ', metrics.mean_squared_error(y_test, predicted2))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, predicted2)))


# In[578]:


#19  ในกรณีนี้ แบบ single linear regression มีประสิทธิภาพมากกว่า SVR แบบ linear

#เพราะมีค่า RMSE ที่น้อยกว่า


# In[579]:


print('RMSE single regression: ', np.sqrt(metrics.mean_squared_error(y_test, predicted)))


# In[580]:


print('RMSE SVR linear: ', np.sqrt(metrics.mean_squared_error(y3_test, predicted3)))


# In[581]:


#20  แบบ simple linear regression


# In[582]:


fig = plt.figure(figsize=(12,8))
plt.scatter(X_test, y_test, color='green', label= 'data')
plt.plot(X_test, predicted, color='red', label= 'Predicted Simple Regression Line')
plt.xlabel('Sqft_living')
plt.ylabel('Price')
plt.legend()


# In[583]:


#20  SVR แบบ linear


# In[587]:


X = df['sqft_living']
y = df['price']


# In[588]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[589]:


X_train = np.array(X_train).reshape(-1,1)
y_train = np.array(y_train).reshape(-1,1)


# In[590]:


from sklearn.preprocessing import StandardScaler
sc_X_train = StandardScaler()
sc_y_train = StandardScaler()

X_train = sc_X_train.fit_transform(X_train)
y_train = sc_y_train.fit_transform(y_train)


# In[591]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'linear')
regressor.fit(X_train, y_train)


# In[592]:


X_test = np.array(X_test).reshape(-1,1)


# In[593]:


sc_X_test = StandardScaler()

X_test = sc_X_test.fit_transform(X_test)


# In[594]:


result = regressor.predict(X_test)
print(result)


# In[595]:


sc_y_train.inverse_transform(result)


# In[596]:


predicted = sc_y_train.inverse_transform(result)
predicted 


# In[598]:


fig = plt.figure(figsize=(12,8))
plt.scatter(sc_X_test.inverse_transform(X_test), y_test, color='green', label= 'data')
plt.plot(sc_X_test.inverse_transform(X_test), predicted , color='red', label= 'Predicted SVR Linear Line')
plt.xlabel('sqft_living')
plt.ylabel('price')
plt.legend()


# In[ ]:




