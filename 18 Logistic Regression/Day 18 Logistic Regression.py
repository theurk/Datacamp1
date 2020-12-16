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


# In[4]:


df = pd.read_csv('train.csv')
df


# In[5]:


#2


# In[6]:


df.head(10)


# In[7]:


df.tail(10)


# In[8]:


df.sample(10)


# In[9]:


#3


# In[10]:


df.info()


# In[11]:


df.describe()


# In[18]:


sns.heatmap(df.isnull(),cbar=False,cmap='Pastel1')


# In[23]:


df = df.drop('Cabin', axis=1)
df


# In[26]:


plt.figure(figsize=(12,10))
sns.heatmap(df.isnull(),cbar=False,cmap='Pastel1')


# In[27]:


avg = df['Age'].mean()
avg


# In[28]:


df['Age'].fillna(value=avg, inplace=True)


# In[29]:


plt.figure(figsize=(12,10))
sns.heatmap(df.isnull(),cbar=False,cmap='Pastel1')


# In[30]:


df.dropna(inplace=True)


# In[31]:


plt.figure(figsize=(12,10))
sns.heatmap(df.isnull(),cbar=False,cmap='Pastel1')


# In[32]:


#4


# In[33]:


df.info()


# In[34]:


df.describe()


# In[35]:


#5


# In[37]:


sns.pairplot(df)


# In[38]:


#6


# In[40]:


fig = plt.figure(figsize=(12,8))
sns.distplot(df['Age'])


# In[41]:


fig = plt.figure(figsize=(12,8))
sns.distplot(df['Pclass'])


# In[42]:


fig = plt.figure(figsize=(12,8))
sns.distplot(df['Fare'])


# In[43]:


fig = plt.figure(figsize=(12,8))
sns.distplot(df['Parch'])


# In[44]:


#7


# In[45]:


sns.heatmap(df.corr())


# In[46]:


#8


# In[47]:


fig = plt.figure(figsize=(12,8))
plt.title('Best correlation')
plt.xlabel('SibSp')
plt.ylabel('Parch')
plt.scatter(df['SibSp'], df['Parch'])


# In[48]:


#9


# In[49]:


fig = plt.figure(figsize=(12,8))
plt.title('Worst correlation')
plt.xlabel('Pclass')
plt.ylabel('Fare')
plt.scatter(df['Pclass'], df['Fare'])


# In[50]:


#10


# In[53]:


fig = plt.figure(figsize=(12,8))
plt.hist(df['Age'],bins=10)
plt.show()


# In[54]:


#11


# In[61]:


fig = plt.figure(figsize=(12,8))
sns.boxplot(x='Pclass',y='Age', data=df)


# In[56]:


#12 - 14


# In[62]:


df.drop(['Name','Ticket'], axis=1, inplace=True)


# In[63]:


df


# In[64]:


sex = pd.get_dummies(df['Sex'])

sex


# In[65]:


sex = pd.get_dummies(df['Sex'], drop_first=True)

sex


# In[66]:


embark = pd.get_dummies(df['Embarked'], drop_first=True)

embark


# In[67]:


df = pd.concat([df,sex,embark],axis=1)
df


# In[68]:


df.drop(['Sex','Embarked'], axis=1, inplace=True)


# In[69]:


df


# In[72]:


fig = plt.figure(figsize=(12,8))
sns.boxplot(x='SibSp',y='Fare', data=df)


# In[80]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# In[202]:


X = df.drop('Survived', axis=1)
X


# In[203]:


y = df['Survived']
y


# In[204]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[167]:


X_train


# In[168]:


#15 แบบเลือกทั้งหมด


# In[169]:


logistic_regression =  LogisticRegression()
logistic_regression.fit(X_train, y_train)


# In[170]:


predicted = logistic_regression.predict(X_test)


# In[ ]:





# In[171]:


#15 แบบเลือกบางตัว


# In[172]:


X2 = df['male']
X2


# In[173]:


y2 = df['Survived']
y2


# In[174]:


X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=100)


# In[175]:


X2_train


# In[176]:


X2_train = np.array(X2_train).reshape(-1,1)


# In[177]:


X2_train


# In[178]:


logistic_regression =  LogisticRegression()
logistic_regression.fit(X2_train, y2_train)


# In[179]:


X2_test = np.array(X2_test).reshape(-1,1)


# In[180]:


predicted2 = logistic_regression.predict(X2_test)


# In[181]:


#16 แบบทั้งหมด


# In[182]:


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


# In[183]:


confusion_matrix(y_test,predicted)


# In[184]:


print('Accuracy = ', accuracy_score(y_test,predicted))
print('F1-score = ', f1_score(y_test,predicted))
print('Precision = ', precision_score(y_test,predicted))
print('Recall = ', recall_score(y_test,predicted))


# In[185]:


#16 แบบเลือก


# In[186]:


confusion_matrix(y2_test,predicted2)


# In[187]:


print('Accuracy = ', accuracy_score(y2_test,predicted2))
print('F1-score = ', f1_score(y2_test,predicted2))
print('Precision = ', precision_score(y2_test,predicted2))
print('Recall = ', recall_score(y2_test,predicted2))


# In[188]:


#17 standardize


# In[205]:


X_train


# In[206]:


y_train


# In[207]:


from sklearn.preprocessing import StandardScaler
sc_X_train = StandardScaler()

X_train = sc_X_train.fit_transform(X_train)


# In[208]:


X_train


# In[209]:


logistic_regression =  LogisticRegression()
logistic_regression.fit(X_train, y_train)


# In[210]:


sc_X_test = StandardScaler()

X_test = sc_X_test.fit_transform(X_test)


# In[211]:


X_test


# In[214]:


predicted3 = logistic_regression.predict(X_test)

print(predicted3)


# In[215]:


#18 แบบทำ standardize


# In[216]:


confusion_matrix(y_test,predicted3)


# In[217]:


print('Accuracy = ', accuracy_score(y_test,predicted3))
print('F1-score = ', f1_score(y_test,predicted3))
print('Precision = ', precision_score(y_test,predicted3))
print('Recall = ', recall_score(y_test,predicted3))


# In[218]:


#18 แบบไม่ทำ standardize


# In[219]:


confusion_matrix(y_test,predicted)


# In[220]:


print('Accuracy = ', accuracy_score(y_test,predicted))
print('F1-score = ', f1_score(y_test,predicted))
print('Precision = ', precision_score(y_test,predicted))
print('Recall = ', recall_score(y_test,predicted))


# In[221]:


#19 เลือก feature ที่สนใจ  เทรนโมเดล แล้ววัดผลเทียบกับข้อ 18


# In[222]:


X2_train


# In[223]:


sc_X2_train = StandardScaler()
sc_X2_test = StandardScaler()

X2_train = sc_X_train.fit_transform(X2_train)
X2_test = sc_X2_test.fit_transform(X2_test)


# In[224]:


X_train


# In[225]:


logistic_regression =  LogisticRegression()
logistic_regression.fit(X2_train, y2_train)


# In[226]:


predicted4 = logistic_regression.predict(X2_test)

print(predicted4)


# In[228]:


confusion_matrix(y2_test,predicted4)


# In[229]:


print('Accuracy = ', accuracy_score(y2_test,predicted4))
print('F1-score = ', f1_score(y2_test,predicted4))
print('Precision = ', precision_score(y2_test,predicted4))
print('Recall = ', recall_score(y2_test,predicted4))


# In[230]:


#20


# In[244]:


X = df.drop('Survived', axis=1)
X


# In[269]:


X3 = df.drop(['Survived','PassengerId','Q','S','Pclass'], axis=1)
X3


# In[270]:


y3 = df['Survived']
y3


# In[271]:


X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=100)


# In[272]:


X3_train


# In[273]:


logistic_regression =  LogisticRegression()
logistic_regression.fit(X3_train, y3_train)


# In[274]:


predicted3 = logistic_regression.predict(X3_test)


# In[275]:


confusion_matrix(y3_test,predicted3)


# In[276]:


print('Accuracy = ', accuracy_score(y3_test,predicted3))
print('F1-score = ', f1_score(y3_test,predicted3))
print('Precision = ', precision_score(y3_test,predicted3))
print('Recall = ', recall_score(y3_test,predicted3))


# In[ ]:





# In[ ]:




