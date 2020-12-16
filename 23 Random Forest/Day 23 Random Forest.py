#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


#1


# In[6]:


df = pd.read_csv('german_credit_data.csv')
df


# In[7]:


#2


# In[8]:


df.head(10)


# In[9]:


df.tail(10)


# In[10]:


df.sample(10)


# In[11]:


#3 + 4


# In[12]:


df.info()


# In[13]:


df.describe()


# In[14]:


sns.heatmap(df.isnull(),cbar=False,cmap='Pastel1')


# In[15]:


df.dropna(inplace=True)


# In[16]:


df.info()


# In[17]:


df = df.drop('Unnamed: 0', axis=1)
df


# In[18]:


#5


# In[19]:


sns.pairplot(df)


# In[20]:


#6


# In[21]:


sns.distplot(df['Age'])


# In[22]:


sns.distplot(df['Credit amount'])


# In[23]:


sns.distplot(df['Duration'])


# In[24]:


#7


# In[25]:


df.corr()


# In[26]:


sns.heatmap(df.corr())


# In[27]:


#8


# In[28]:


plt.title('Best correlation')
plt.xlabel('Duration')
plt.ylabel('Credit amount')
plt.scatter(df['Duration'], df['Credit amount'])


# In[29]:


#9


# In[30]:


fig = plt.figure(figsize=(12,8))
plt.title('Worst correlation')
plt.xlabel('Duration')
plt.ylabel('Age')
plt.scatter(df['Duration'], df['Age'])


# In[31]:


#10


# In[32]:


fig = plt.figure(figsize=(12,8))
plt.hist(df['Age'],bins=10)
plt.show()


# In[33]:


#11


# In[34]:


fig = plt.figure(figsize=(12,8))
sns.boxplot(x='Sex',y='Age', data=df)


# In[35]:


#12


# In[36]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, df['Risk'], test_size=0.2, random_state=100)


# In[37]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[38]:


#13


# In[39]:


fig = plt.figure(figsize=(12,8))
sns.countplot(x='Sex', data=df, palette='rainbow_r')


# In[40]:


fig = plt.figure(figsize=(12,8))
sns.countplot(x='Housing', data=df, palette='rainbow_r')


# In[41]:


fig = plt.figure(figsize=(12,8))
sns.countplot(x='Saving accounts', data=df, palette='rainbow_r')


# In[42]:


fig = plt.figure(figsize=(12,8))
sns.countplot(x='Purpose', data=df, palette='rainbow_r')


# In[43]:


fig = plt.figure(figsize=(12,8))
sns.countplot(x='Checking account', data=df, palette='rainbow_r')


# In[44]:


fig = plt.figure(figsize=(12,8))
sns.countplot(x='Risk', data=df, palette='rainbow_r')


# In[45]:


#14 ทำแบบ default พอ  เพราะข้อมูลส่วนใหญ่เป็นแบบ catergorical


# In[46]:


#15 เทรนโมเดล แต่ต้องจัดการข้อมูลประเภท catergorical ก่อน


# In[47]:


df_real = pd.get_dummies(df,drop_first=True)

df_real


# In[53]:


#เทรนโมเดล แบบ Default

X = df_real.drop(['Risk_good'], axis=1)
y = df_real['Risk_good']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[54]:


X_train


# In[56]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[60]:


from sklearn.ensemble import RandomForestClassifier


# In[61]:


rf = RandomForestClassifier()


# In[62]:


rf.fit(X_train, y_train)


# In[63]:


predicted = rf.predict(X_test)
predicted


# In[64]:


#16


# In[65]:


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


# In[66]:


#default all features

confusion_matrix(y_test,predicted)


# In[67]:


#default all features

print('Accuracy = ', accuracy_score(y_test,predicted))
print('F1-score = ', f1_score(y_test,predicted))
print('Precision = ', precision_score(y_test,predicted))
print('Recall = ', recall_score(y_test,predicted))


# In[68]:


#17


# In[69]:


from sklearn.model_selection import GridSearchCV


# In[76]:


param_combination = {'max_depth':[4,8,16,32,64,128,256], 
                     'min_samples_leaf':[1,2,4,8,12,16,20],
                     'n_estimators':[10,20,50,100,500]}


# In[77]:


grid_search = GridSearchCV(RandomForestClassifier(), param_combination, verbose=1)


# In[78]:


grid_search.fit(X_train, y_train)


# In[79]:


grid_search.best_params_


# In[80]:


grid_search.best_estimator_


# In[81]:


grid_predicted = grid_search.predict(X_test)
grid_predicted


# In[82]:


#default grid_search

confusion_matrix(y_test,grid_predicted)


# In[83]:


#default grid_search

print('Accuracy = ', accuracy_score(y_test,grid_predicted))
print('F1-score = ', f1_score(y_test,grid_predicted))
print('Precision = ', precision_score(y_test,grid_predicted))
print('Recall = ', recall_score(y_test,grid_predicted))


# In[84]:


#18


# In[85]:


df_real


# In[86]:


df_real.corr()


# In[87]:


X2 = df_real[['Credit amount','Duration']]

X2


# In[88]:


X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=100)


# In[89]:


X_train


# In[90]:


rf2 = RandomForestClassifier()


# In[91]:


rf2.fit(X_train, y_train)


# In[92]:


predicted2 = rf2.predict(X_test)
predicted2


# In[93]:


#default selected feature

confusion_matrix(y_test,predicted2)


# In[94]:


#default selected feature

print('Accuracy = ', accuracy_score(y_test,predicted2))
print('F1-score = ', f1_score(y_test,predicted2))
print('Precision = ', precision_score(y_test,predicted2))
print('Recall = ', recall_score(y_test,predicted2))


# In[95]:


#19 - 21 Visualization


# In[96]:


# แต่ทำ normalize ก่อน ยังไม่ได้ทำ


# In[97]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[98]:


from sklearn.preprocessing import MinMaxScaler


# In[99]:


min_max_scaler = MinMaxScaler()
min_max_scaler2 = MinMaxScaler()


# In[100]:


X_train = min_max_scaler.fit_transform(X_train)

X_train


# In[101]:


X_train = pd.DataFrame(X_train, columns=X.columns)

X_train


# In[152]:


X_test = min_max_scaler2.fit_transform(X_test)

X_test


# In[153]:


X_test = pd.DataFrame(X_test, columns=X.columns)

X_test


# In[102]:


rf3 = RandomForestClassifier()


# In[103]:


rf3.fit(X_train, y_train)


# In[104]:


predicted3 = rf3.predict(X_test)
predicted3


# In[105]:


#Normalized all features

confusion_matrix(y_test,predicted3)


# In[106]:


#Normalized all features

print('Accuracy = ', accuracy_score(y_test,predicted3))
print('F1-score = ', f1_score(y_test,predicted3))
print('Precision = ', precision_score(y_test,predicted3))
print('Recall = ', recall_score(y_test,predicted3))


# In[107]:


Score = pd.DataFrame({
    'Type': ['Default','Grid Search','Normalization'],
    'F1 Score' : [f1_score(y_test,predicted),f1_score(y_test,grid_predicted),f1_score(y_test,predicted3)],
    'Recall': [recall_score(y_test,predicted),recall_score(y_test,grid_predicted),recall_score(y_test,predicted3)],
    'Accuracy': [accuracy_score(y_test,predicted),accuracy_score(y_test,grid_predicted),accuracy_score(y_test,predicted3)]})


# In[108]:


Score


# In[109]:


Score.set_index('Type')


# In[110]:


Score = Score.set_index('Type')


# In[111]:


Score.plot(kind='bar')
plt.show()


# In[112]:


#22 ลอง Default all features + Normalize + Grid search


# In[113]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[114]:


X_train = min_max_scaler.fit_transform(X_train)

X_train


# In[115]:


X_train = pd.DataFrame(X_train, columns=X.columns)

X_train


# In[116]:


X_test = min_max_scaler2.fit_transform(X_test)

X_test


# In[117]:


X_test = pd.DataFrame(X_test, columns=X.columns)

X_test


# In[118]:


param_combination = {'max_depth':[4,8,16,32,64,128,256], 
                     'min_samples_leaf':[1,2,4,8,12,16,20],
                     'n_estimators':[10,20,50,100,500]}


# In[119]:


grid_search2 = GridSearchCV(RandomForestClassifier(), param_combination, verbose=1)


# In[120]:


grid_search2.fit(X_train, y_train)


# In[121]:


grid_search2.best_params_


# In[122]:


grid_search2.best_estimator_


# In[123]:


grid_predicted2 = grid_search2.predict(X_test)
grid_predicted2


# In[124]:


#Normalized + grid_search

confusion_matrix(y_test,grid_predicted2)


# In[125]:


#Normalized + grid_search  ผลลัพธ์แย่กว่าเก่า

print('Accuracy = ', accuracy_score(y_test,grid_predicted2))
print('F1-score = ', f1_score(y_test,grid_predicted2))
print('Precision = ', precision_score(y_test,grid_predicted2))
print('Recall = ', recall_score(y_test,grid_predicted2))


# In[126]:


#ลองแบบเลือก feature มาบางอันพอ  + Grid Search


# In[127]:


X2 = df_real[['Credit amount','Duration']]

X2


# In[128]:


X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=100)


# In[129]:


grid_search3 = GridSearchCV(RandomForestClassifier(), param_combination, verbose=1)


# In[130]:


grid_search3.fit(X_train, y_train)


# In[131]:


grid_search3.best_params_


# In[132]:


grid_search3.best_estimator_


# In[133]:


grid_predicted3 = grid_search3.predict(X_test)
grid_predicted3


# In[134]:


#เลือกบาง features + grid_search

confusion_matrix(y_test,grid_predicted3)


# In[135]:


#เลือกบาง features + grid_search   ได้ผลลัพธ์ดีที่สุด

print('Accuracy = ', accuracy_score(y_test,grid_predicted3))
print('F1-score = ', f1_score(y_test,grid_predicted3))
print('Precision = ', precision_score(y_test,grid_predicted3))
print('Recall = ', recall_score(y_test,grid_predicted3))


# In[136]:


#23  สร้าง Barchart เทียบระหว่าง DecisionTree กับ RandomForest ที่ดีที่สุด


# In[142]:


Score = pd.DataFrame({
    'Type': ['Decision Tree','Random Forest'],
    'F1 Score' : [0.8169014084507042,f1_score(y_test,grid_predicted3)],
    'Recall': [0.8923076923076924,recall_score(y_test,grid_predicted3)],
    'Accuracy': [0.7523809523809524,accuracy_score(y_test,grid_predicted3)],
    'Precision': [0.7532467532467533,precision_score(y_test,grid_predicted3)]})


# In[143]:


Score


# In[144]:


Score = Score.set_index('Type')

Score


# In[145]:


Score.plot(kind='bar')
plt.show()


# In[ ]:




