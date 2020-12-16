#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[47]:


#1


# In[48]:


df = pd.read_csv('Wine_completed.csv')
df


# In[49]:


#2


# In[50]:


df.head(10)


# In[51]:


df.tail(10)


# In[52]:


df.sample(10)


# In[53]:


#3 + 4


# In[54]:


df.info()


# In[55]:


df.describe()


# In[56]:


#5


# In[62]:


sns.pairplot(df)

#6
# In[63]:


sns.distplot(df['Alcohol'])


# In[64]:


sns.distplot(df['Malic acid'])


# In[65]:


sns.distplot(df['Ash'])


# In[66]:


sns.distplot(df['Magnesium'])


# In[67]:


#7


# In[68]:


df.corr()


# In[69]:


sns.heatmap(df.corr())


# In[70]:


#8


# In[71]:


#fig = plt.figure(figsize=(12,8))
plt.title('Best correlation')
plt.xlabel('Flavanoids')
plt.ylabel('Total penols')
plt.scatter(df['Flavanoids'], df['Total penols'])


# In[72]:


#9


# In[73]:


fig = plt.figure(figsize=(12,8))
plt.title('Worst correlation')
plt.xlabel('Flavanoids')
plt.ylabel('Class')
plt.scatter(df['Flavanoids'], df['Class'])


# In[74]:


#10


# In[75]:


fig = plt.figure(figsize=(12,8))
plt.hist(df['Flavanoids'],bins=10)
plt.show()


# In[76]:


#11


# In[77]:


fig = plt.figure(figsize=(12,8))
sns.boxplot(x='Class',y='Flavanoids', data=df)


# In[78]:


#12


# In[91]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


# In[95]:


X = df.drop('Class', axis=1)
X


# In[96]:


y = df['Class']
y


# In[97]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[98]:


X_train


# In[101]:


#13


# In[102]:


sns.distplot(df['Flavanoids'])


# In[ ]:


#14 พิจารณาแล้วว่า ไม่จำเป็นต้องทำ normalization หรือ standardization

#เพราะว่า ข้อมูล y = df['Class']  มีค่าเท่ากับ 1,2,3  ไม่ใช่ 0 กับ 1


# In[103]:


df


# In[104]:


#15


# In[105]:


from sklearn.neighbors import KNeighborsClassifier


# In[106]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[107]:


knn.fit(X_train,y_train)


# In[108]:


predicted = knn.predict(X_test)


# In[109]:


#16


# In[110]:


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


# In[111]:


confusion_matrix(y_test,predicted)


# In[113]:


print('Accuracy = ', accuracy_score(y_test,predicted))
print('F1-score = ', f1_score(y_test,predicted, average='micro'))
print('Precision = ', precision_score(y_test,predicted, average='micro'))
print('Recall = ', recall_score(y_test,predicted, average='micro'))


# In[114]:


#17 หาค่า k ที่ดีที่สุด


# In[115]:


accuracy_lst = []

for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    predicted_i = knn.predict(X_test)
    accuracy_lst.append(accuracy_score(y_test, predicted_i))


# In[116]:


accuracy_lst


# In[117]:


# ค่า k ที่ดีที่สุดคือ k = 1


# In[118]:


#18


# In[119]:


df


# In[120]:


X2 = df['Nonflavanoids penols']
X2


# In[121]:


y2 = df['Class']
y2


# In[122]:


X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=100)


# In[123]:


X2_train


# In[124]:


X2_train = np.array(X2_train).reshape(-1,1)


# In[125]:


X2_train


# In[126]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X2_train,y2_train)


# In[127]:


X2_test = np.array(X2_test).reshape(-1,1)


# In[128]:


predicted2 = knn.predict(X2_test)


# In[129]:


confusion_matrix(y2_test,predicted2)


# In[131]:


print('Accuracy = ', accuracy_score(y2_test,predicted2))
print('F1-score = ', f1_score(y2_test,predicted2, average='micro'))
print('Precision = ', precision_score(y2_test,predicted2, average='micro'))
print('Recall = ', recall_score(y2_test,predicted2, average='micro'))


# In[134]:


accuracy_lst2 = []

for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X2_train,y2_train)
    predicted2_i = knn.predict(X2_test)
    accuracy_lst2.append(accuracy_score(y2_test, predicted2_i))


# In[135]:


accuracy_lst2


# In[136]:


#สรุป คือ เทรนโมเดลแบบเลือกทุก features มีความแม่นยำมากกว่า


# In[137]:


#19


# In[138]:


plt.figure(figsize=(10,8))
plt.plot(range(1,50), accuracy_lst2, color='black', linestyle='dashed',
                       marker='o', markerfacecolor='blue', markersize=7)
plt.xlabel('K')
plt.ylabel('Accuracy')


# In[139]:


#20


# In[140]:


df


# In[141]:


df.corr()


# In[142]:


sns.heatmap(df.corr())


# In[148]:


X3 = df[['Flavanoids','OD280/OD315 of diluted wines','Total penols']]
X3


# In[149]:


y3 = df['Class']
y3


# In[150]:


X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=100)


# In[151]:


X3_train


# In[153]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X3_train,y3_train)


# In[154]:


predicted3 = knn.predict(X3_test)


# In[155]:


confusion_matrix(y3_test,predicted3)


# In[156]:


print('Accuracy = ', accuracy_score(y3_test,predicted3))
print('F1-score = ', f1_score(y3_test,predicted3, average='micro'))
print('Precision = ', precision_score(y3_test,predicted3, average='micro'))
print('Recall = ', recall_score(y3_test,predicted3, average='micro'))


# In[157]:


accuracy_lst3 = []

for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X3_train,y3_train)
    predicted3_i = knn.predict(X3_test)
    accuracy_lst3.append(accuracy_score(y3_test, predicted3_i))


# In[158]:


accuracy_lst3


# In[159]:


plt.figure(figsize=(10,8))
plt.plot(range(1,50), accuracy_lst3, color='black', linestyle='dashed',
                       marker='o', markerfacecolor='blue', markersize=7)
plt.xlabel('K')
plt.ylabel('Accuracy')


# In[160]:


#สรุป ถ้า k = 5 จะได้ Accuracy สูงถึง 86%


# In[161]:


knn = KNeighborsClassifier(n_neighbors=5)


# In[162]:


knn.fit(X3_train,y3_train)
predicted3 = knn.predict(X3_test)


# In[163]:


confusion_matrix(y3_test,predicted3)


# In[164]:


print('Accuracy = ', accuracy_score(y3_test,predicted3))
print('F1-score = ', f1_score(y3_test,predicted3, average='micro'))
print('Precision = ', precision_score(y3_test,predicted3, average='micro'))
print('Recall = ', recall_score(y3_test,predicted3, average='micro'))


# In[ ]:




