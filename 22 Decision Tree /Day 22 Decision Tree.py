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


# In[15]:


df = pd.read_csv('german_credit_data.csv')
df


# In[16]:


#2


# In[17]:


df.head(10)


# In[18]:


df.tail(10)


# In[19]:


df.sample(10)


# In[20]:


#3 + 4


# In[21]:


df.info()


# In[22]:


df.describe()


# In[23]:


sns.heatmap(df.isnull(),cbar=False,cmap='Pastel1')


# In[24]:


df.dropna(inplace=True)


# In[25]:


df.info()


# In[26]:


df = df.drop('Unnamed: 0', axis=1)
df


# In[27]:


#5


# In[28]:


sns.pairplot(df)


# In[29]:


#6


# In[30]:


sns.distplot(df['Age'])


# In[31]:


sns.distplot(df['Credit amount'])


# In[32]:


sns.distplot(df['Duration'])


# In[33]:


#7


# In[35]:


df.corr()


# In[37]:


sns.heatmap(df.corr())


# In[38]:


#8


# In[39]:


plt.title('Best correlation')
plt.xlabel('Duration')
plt.ylabel('Credit amount')
plt.scatter(df['Duration'], df['Credit amount'])


# In[40]:


#9


# In[41]:


fig = plt.figure(figsize=(12,8))
plt.title('Worst correlation')
plt.xlabel('Duration')
plt.ylabel('Age')
plt.scatter(df['Duration'], df['Age'])


# In[42]:


#10


# In[43]:


fig = plt.figure(figsize=(12,8))
plt.hist(df['Age'],bins=10)
plt.show()


# In[44]:


#11


# In[45]:


fig = plt.figure(figsize=(12,8))
sns.boxplot(x='Sex',y='Age', data=df)


# In[46]:


#12


# In[47]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, df['Risk'], test_size=0.2, random_state=100)


# In[48]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[49]:


#13


# In[52]:


fig = plt.figure(figsize=(12,8))
sns.countplot(x='Sex', data=df, palette='rainbow_r')


# In[53]:


fig = plt.figure(figsize=(12,8))
sns.countplot(x='Housing', data=df, palette='rainbow_r')


# In[54]:


fig = plt.figure(figsize=(12,8))
sns.countplot(x='Saving accounts', data=df, palette='rainbow_r')


# In[55]:


fig = plt.figure(figsize=(12,8))
sns.countplot(x='Purpose', data=df, palette='rainbow_r')


# In[56]:


fig = plt.figure(figsize=(12,8))
sns.countplot(x='Checking account', data=df, palette='rainbow_r')


# In[57]:


fig = plt.figure(figsize=(12,8))
sns.countplot(x='Risk', data=df, palette='rainbow_r')


# In[92]:


#14 ทำแบบ default พอ  เพราะข้อมูลส่วนใหญ่เป็นแบบ catergorical


# In[93]:


#15 เทรนโมเดล แต่ต้องจัดการข้อมูลประเภท catergorical ก่อน


# In[94]:


df_real = pd.get_dummies(df,drop_first=True)

df_real


# In[95]:


#เทรนโมเดล แบบ Default

X = df_real.drop(['Risk_good'], axis=1)
y = df_real['Risk_good']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[96]:


X_train


# In[97]:


from sklearn.tree import DecisionTreeClassifier


# In[98]:


dtree = DecisionTreeClassifier()


# In[99]:


dtree.fit(X_train, y_train)


# In[101]:


predicted = dtree.predict(X_test)
predicted


# In[102]:


#16


# In[103]:


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


# In[104]:


#default all features

confusion_matrix(y_test,predicted)


# In[105]:


#default all features

print('Accuracy = ', accuracy_score(y_test,predicted))
print('F1-score = ', f1_score(y_test,predicted))
print('Precision = ', precision_score(y_test,predicted))
print('Recall = ', recall_score(y_test,predicted))


# In[106]:


#17


# In[115]:


from sklearn.model_selection import GridSearchCV


# In[116]:


param_combination = {'max_depth':[4,8,16,32,64,128,256], 'min_samples_leaf':[1,2,4,8,12,16,20]}


# In[117]:


grid_search = GridSearchCV(DecisionTreeClassifier(), param_combination, verbose=1)


# In[118]:


grid_search.fit(X_train, y_train)


# In[119]:


grid_search.best_params_


# In[120]:


grid_search.best_estimator_


# In[121]:


grid_predicted = grid_search.predict(X_test)
grid_predicted


# In[122]:


#default grid_search

confusion_matrix(y_test,grid_predicted)


# In[123]:


#default grid_search

print('Accuracy = ', accuracy_score(y_test,grid_predicted))
print('F1-score = ', f1_score(y_test,grid_predicted))
print('Precision = ', precision_score(y_test,grid_predicted))
print('Recall = ', recall_score(y_test,grid_predicted))


# In[124]:


#18


# In[125]:


df_real


# In[128]:


df_real.corr()


# In[131]:


X2 = df_real[['Credit amount','Duration']]

X2


# In[132]:


X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=100)


# In[133]:


X_train


# In[134]:


dtree2 = DecisionTreeClassifier()


# In[136]:


dtree2.fit(X_train, y_train)


# In[137]:


predicted2 = dtree2.predict(X_test)
predicted2


# In[138]:


#default selected feature

confusion_matrix(y_test,predicted2)


# In[139]:


#default selected feature

print('Accuracy = ', accuracy_score(y_test,predicted2))
print('F1-score = ', f1_score(y_test,predicted2))
print('Precision = ', precision_score(y_test,predicted2))
print('Recall = ', recall_score(y_test,predicted2))


# In[140]:


#19 - 21 Visualization


# In[141]:


# แต่ทำ normalize ก่อน ยังไม่ได้ทำ


# In[147]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[148]:


from sklearn.preprocessing import MinMaxScaler


# In[149]:


min_max_scaler = MinMaxScaler()
min_max_scaler2 = MinMaxScaler()


# In[150]:


X_train = min_max_scaler.fit_transform(X_train)

X_train


# In[151]:


X_train = pd.DataFrame(X_train, columns=X.columns)

X_train


# In[152]:


X_test = min_max_scaler2.fit_transform(X_test)

X_test


# In[153]:


X_test = pd.DataFrame(X_test, columns=X.columns)

X_test


# In[154]:


dtree3 = DecisionTreeClassifier()


# In[155]:


dtree3.fit(X_train, y_train)


# In[156]:


predicted3 = dtree3.predict(X_test)
predicted3


# In[157]:


#Normalized all features

confusion_matrix(y_test,predicted3)


# In[158]:


#Normalized all features

print('Accuracy = ', accuracy_score(y_test,predicted3))
print('F1-score = ', f1_score(y_test,predicted3))
print('Precision = ', precision_score(y_test,predicted3))
print('Recall = ', recall_score(y_test,predicted3))


# In[160]:


Score = pd.DataFrame({
    'Type': ['Default','Grid Search','Normalization'],
    'F1 Score' : [f1_score(y_test,predicted),f1_score(y_test,grid_predicted),f1_score(y_test,predicted3)],
    'Recall': [recall_score(y_test,predicted),recall_score(y_test,grid_predicted),recall_score(y_test,predicted3)],
    'Accuracy': [accuracy_score(y_test,predicted),accuracy_score(y_test,grid_predicted),accuracy_score(y_test,predicted3)]})


# In[161]:


Score


# In[162]:


Score.set_index('Type')


# In[164]:


Score = Score.set_index('Type')


# In[165]:


Score.plot(kind='bar')
plt.show()


# In[224]:


#22 ลอง Default all features + Normalize + Grid search


# In[225]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[228]:


X_train = min_max_scaler.fit_transform(X_train)

X_train


# In[229]:


X_train = pd.DataFrame(X_train, columns=X.columns)

X_train


# In[230]:


X_test = min_max_scaler2.fit_transform(X_test)

X_test


# In[231]:


X_test = pd.DataFrame(X_test, columns=X.columns)

X_test


# In[232]:


param_combination = {'max_depth':[4,8,16,32,64,128,256], 'min_samples_leaf':[1,2,4,8,12,16,20]}


# In[233]:


grid_search2 = GridSearchCV(DecisionTreeClassifier(), param_combination, verbose=1)


# In[234]:


grid_search2.fit(X_train, y_train)


# In[237]:


grid_search2.best_params_


# In[238]:


grid_search2.best_estimator_


# In[239]:


grid_predicted2 = grid_search2.predict(X_test)
grid_predicted2


# In[240]:


#Normalized + grid_search

confusion_matrix(y_test,grid_predicted2)


# In[241]:


#Normalized + grid_search  ผลลัพธ์แย่กว่าเก่า

print('Accuracy = ', accuracy_score(y_test,grid_predicted2))
print('F1-score = ', f1_score(y_test,grid_predicted2))
print('Precision = ', precision_score(y_test,grid_predicted2))
print('Recall = ', recall_score(y_test,grid_predicted2))


# In[242]:


#ลองแบบเลือก feature มาบางอันพอ  + Grid Search


# In[243]:


X2 = df_real[['Credit amount','Duration']]

X2


# In[244]:


X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=100)


# In[245]:


grid_search3 = GridSearchCV(DecisionTreeClassifier(), param_combination, verbose=1)


# In[246]:


grid_search3.fit(X_train, y_train)


# In[247]:


grid_search3.best_params_


# In[248]:


grid_search3.best_estimator_


# In[249]:


grid_predicted3 = grid_search3.predict(X_test)
grid_predicted3


# In[250]:


#เลือกบาง features + grid_search

confusion_matrix(y_test,grid_predicted3)


# In[251]:


#เลือกบาง features + grid_search   ได้ผลลัพธ์ดีที่สุด

print('Accuracy = ', accuracy_score(y_test,grid_predicted3))
print('F1-score = ', f1_score(y_test,grid_predicted3))
print('Precision = ', precision_score(y_test,grid_predicted3))
print('Recall = ', recall_score(y_test,grid_predicted3))


# In[ ]:




