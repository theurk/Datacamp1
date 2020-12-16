#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # 1

# In[3]:


df = pd.read_csv('Churn_Modelling.csv')
df


# # 2

# In[4]:


df.head(10)


# In[5]:


df.tail(10)


# In[6]:


df.sample(10)


# # 3

# In[7]:


df.info()


# In[8]:


df.describe()


# # 4

# In[9]:


sns.pairplot(df)


# # 5

# In[11]:


df['Age'].unique()


# In[12]:


df['Geography'].unique()


# In[13]:


df['Gender'].unique()


# In[14]:


df['Surname'].unique()


# In[27]:


df.nunique()


# # 6

# In[17]:


df = df.drop(['RowNumber','CustomerId'], axis=1)

df


# # 7

# In[18]:


sns.distplot(df['CreditScore'])


# In[19]:


sns.distplot(df['Age'])


# In[20]:


sns.distplot(df['Tenure'])


# In[21]:


sns.distplot(df['Balance'])


# In[22]:


sns.distplot(df['EstimatedSalary'])


# # 8

# In[31]:


df[df['Balance']!=0]


# In[32]:


df[df['Balance']!=0]['Balance']


# In[33]:


df_balance = df[df['Balance']!=0]['Balance']

df_balance


# In[34]:


sns.distplot(df_balance)


# # 9

# In[36]:


fig = plt.figure(figsize=(12,8))
sns.countplot(x='Geography', data=df, palette='rainbow_r')


# # 10

# In[37]:


fig = plt.figure(figsize=(12,8))
sns.countplot(x='Gender', data=df, palette='rainbow_r')


# # 11

# In[66]:


df['Surname'].value_counts()


# # 12

# In[67]:


df_surname = df['Surname'].value_counts()

df_surname


# In[68]:


df_surname[df_surname > 1]


# # 13

# In[69]:


df_surname.head(15)


# In[70]:


index = df_surname.head(15).index
index


# In[71]:


values = df_surname.head(15).values
values


# In[72]:


fig = plt.figure(figsize=(12,8))
sns.barplot(index, values)


# In[73]:


df_surname.head(15)[::-1]


# In[74]:


index = df_surname.head(15)[::-1].index
index


# In[75]:


values = df_surname.head(15)[::-1].values
values


# In[76]:


fig = plt.figure(figsize=(12,8))
sns.barplot(index, values)


# # 14

# In[77]:


df.corr()


# # 15

# In[84]:


fig = plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linecolor='white', linewidth=2)


# # 16

# In[86]:


fig = plt.figure(figsize=(12,8))
sns.scatterplot(x='Balance', y='EstimatedSalary',palette='plasma', data=df)


# # 17

# In[89]:


df_no_zero_balance = df[df['Balance']!=0]
df_no_zero_balance


# In[91]:


df_no_zero_balance.corr()


# In[92]:


fig = plt.figure(figsize=(12,8))
sns.scatterplot(x='Balance', y='EstimatedSalary',palette='plasma', data=df_no_zero_balance)


# # 18

# In[93]:


fig = plt.figure(figsize=(12,8))
sns.scatterplot(x='Age', y='EstimatedSalary',palette='plasma', data=df)


# # 19

# In[94]:


fig = plt.figure(figsize=(12,8))
sns.countplot(x='Geography', data=df, palette='rainbow_r',hue='Gender')


# # 20

# In[95]:


df[df['Geography']=='France']


# In[96]:


df[df['Geography']=='France']['Gender'].value_counts()


# In[97]:


df[df['Geography']=='Spain']['Gender'].value_counts()


# In[98]:


df[df['Geography']=='Germany']['Gender'].value_counts()


# # 21

# In[100]:


df['Exited'].value_counts()

# (2037/7963+2037)*100 = 20.37%


# # 22

# In[101]:


import plotly.express as px


# In[102]:


df.head(5)


# In[106]:


fig = px.pie(df, values='Exited', names='Geography', title='Sum of exited customers per geography')
fig.show()


#chart อันนี้ ยังบอกข้อมูลได้ไม่ดีเท่าไหร่ เพราะว่า ยังไม่รู้อัตราส่วนต่อประชากร


# # 23

# In[108]:


df.groupby('Geography').sum()


# In[114]:


values = df.groupby('Geography').sum()['Exited']
values


# In[115]:


index = df.groupby('Geography').sum().index
index


# In[116]:


fig = px.bar(df, y=values, x=index, title='Sum of exited customers per geography')
fig.show()


# # 24

# In[117]:


df.groupby('Geography').mean()


# In[118]:


df[df['Geography']=='Germany']['Exited'].value_counts()


# In[119]:


df[df['Geography']=='France']['Exited'].value_counts()


# # 25

# In[121]:


df_geo_mean = df.groupby('Geography').mean()


# In[122]:


fig = px.pie(df_geo_mean, values='Exited', names=df_geo_mean.index, title='Mean of exited customers per geography')
fig.show()


# # 26

# In[124]:


df[df['Exited']==1]


# In[125]:


df[df['Exited']==1]['Age']


# In[126]:


sns.distplot(df[df['Exited']==1]['Age'])


# # 27

# In[129]:


df_groupby_age = df.groupby('Age').mean().head(10)
df_groupby_age


# In[131]:


fig = px.scatter(df_groupby_age, x=df_groupby_age.index, y='EstimatedSalary',
                 size='Balance', color='CreditScore')

fig.show()


# # 28

# In[134]:


df_num =df.groupby('NumOfProducts').mean()
df_num


# In[136]:


fig = px.bar(df_num, y='Exited', x=df_num.index, title='% of charn vs num of products')
fig.show()


# # 29

# In[139]:


df_gender =df.groupby('Gender').mean()
df_gender


# In[141]:


fig = px.bar(df_gender, y='Exited', x=df_gender.index, title='% of charn vs Gender')
fig.show()


# # 30

# In[142]:


#ใช้ boxplot ตรวจดู Outlier


# In[143]:


df


# In[145]:


fig = plt.figure(figsize=(12,8))
px.box(df, y='CreditScore')


# In[146]:


fig = plt.figure(figsize=(12,8))
px.box(df, y='Age')


# In[147]:


fig = plt.figure(figsize=(12,8))
px.box(df, y='Tenure')


# In[148]:


fig = plt.figure(figsize=(12,8))
px.box(df, y='Balance')


# In[149]:


fig = plt.figure(figsize=(12,8))
px.box(df, y='EstimatedSalary')


# # 31

# In[150]:


df['Age'] = [62 if x > 62 else x for x in df['Age']]

fig = plt.figure(figsize=(12,8))
px.box(df, y='Age')


# In[151]:


df['CreditScore'] = [383 if x < 383 else x for x in df['CreditScore']]

fig = plt.figure(figsize=(12,8))
px.box(df, y='CreditScore')


# # 32

# In[152]:


df = df.rename(columns={'Exited':'Churn'})

df


# # 33

# In[156]:


df = df.drop(['Surname'], axis=1)

df


# # 34

# In[158]:


df_real = pd.get_dummies(df, drop_first=True)

df_real


# # 35

# In[159]:


from sklearn.model_selection import train_test_split


# In[162]:


X = df_real.drop(['Churn'], axis=1)
X


# In[163]:


y = df_real['Churn']
y


# In[164]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[165]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # 36  LogisticRegression

# In[167]:


from sklearn.linear_model import LogisticRegression


# In[168]:


logistic_regression = LogisticRegression()


# In[169]:


logistic_regression.fit(X_train, y_train)


# In[170]:


logistic_predicted = logistic_regression.predict(X_test)


# # 37

# In[171]:


fig = plt.figure(figsize=(12,8))
sns.countplot(logistic_predicted)


# # 38

# In[172]:


from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score


# In[173]:


confusion_matrix(y_test, logistic_predicted)


# In[186]:


print('Logistic Regression Result')

print('Accuracy: ', accuracy_score(y_test, logistic_predicted))
print('F1 Score: ', f1_score(y_test, logistic_predicted))
print('Precision: ', precision_score(y_test, logistic_predicted))
print('Recall: ', recall_score(y_test, logistic_predicted))


# # 39   KNeighborsClassifier

# In[178]:


from sklearn.neighbors import KNeighborsClassifier


# In[180]:


knn = KNeighborsClassifier()


# In[181]:


knn.fit(X_train, y_train)


# In[182]:


knn_predicted = knn.predict(X_test)


# # 40

# In[183]:


fig = plt.figure(figsize=(12,8))
sns.countplot(knn_predicted)


# In[184]:


confusion_matrix(y_test, knn_predicted)


# # 41

# In[187]:


print('KNeighborsClassifier Result')

print('Accuracy: ', accuracy_score(y_test, knn_predicted))
print('F1 Score: ', f1_score(y_test, knn_predicted))
print('Precision: ', precision_score(y_test, knn_predicted))
print('Recall: ', recall_score(y_test, knn_predicted))


# # 42  Support Vector Machine

# In[190]:


from sklearn.svm import SVC


# In[191]:


svc = SVC()


# In[192]:


svc.fit(X_train, y_train)


# In[195]:


svm_predicted = svc.predict(X_test)


# # 43

# In[196]:


fig = plt.figure(figsize=(12,8))
sns.countplot(svm_predicted)


# # 44

# In[197]:


confusion_matrix(y_test, svm_predicted)


# In[199]:


print('Support Vector Machine Result')

print('Accuracy: ', accuracy_score(y_test, svm_predicted))
print('F1 Score: ', f1_score(y_test, svm_predicted))
print('Precision: ', precision_score(y_test, svm_predicted))
print('Recall: ', recall_score(y_test, svm_predicted))


# # 45

# In[200]:


#เนื่องจากค่า True Positive มีค่าเป็น 0 เลยทำให้ทั้งสามค่า f1 score, precision, recall มีค่าเป็น 0


# # 46  Naive Bayes

# In[201]:


from sklearn.naive_bayes import GaussianNB


# In[202]:


nb = GaussianNB()


# In[203]:


nb.fit(X_train, y_train)


# In[204]:


nb_predicted = nb.predict(X_test)


# # 47

# In[205]:


fig = plt.figure(figsize=(12,8))
sns.countplot(nb_predicted)


# # 48
# 

# In[206]:


confusion_matrix(y_test, nb_predicted)


# In[207]:


print('Naive Bayes Result')

print('Accuracy: ', accuracy_score(y_test, nb_predicted))
print('F1 Score: ', f1_score(y_test, nb_predicted))
print('Precision: ', precision_score(y_test, nb_predicted))
print('Recall: ', recall_score(y_test, nb_predicted))


# # 49  Decision Tree

# In[209]:


from sklearn.tree import DecisionTreeClassifier


# In[210]:


dt = DecisionTreeClassifier()


# In[211]:


dt.fit(X_train, y_train)


# In[212]:


dt_predicted = dt.predict(X_test)


# # 50

# In[213]:


fig = plt.figure(figsize=(12,8))
sns.countplot(dt_predicted)


# # 51

# In[214]:


confusion_matrix(y_test, dt_predicted)


# In[215]:


print('Decision Tree Result')

print('Accuracy: ', accuracy_score(y_test, dt_predicted))
print('F1 Score: ', f1_score(y_test, dt_predicted))
print('Precision: ', precision_score(y_test, dt_predicted))
print('Recall: ', recall_score(y_test, dt_predicted))


# # 52 Random Forest

# In[218]:


from sklearn.ensemble import RandomForestClassifier


# In[219]:


rf = RandomForestClassifier()


# In[220]:


rf.fit(X_train, y_train)


# In[221]:


rf_predicted = rf.predict(X_test)


# # 53

# In[222]:


fig = plt.figure(figsize=(12,8))
sns.countplot(rf_predicted)


# # 54

# In[223]:


confusion_matrix(y_test, rf_predicted)


# In[224]:


print('Random Forest Result')

print('Accuracy: ', accuracy_score(y_test, rf_predicted))
print('F1 Score: ', f1_score(y_test, rf_predicted))
print('Precision: ', precision_score(y_test, rf_predicted))
print('Recall: ', recall_score(y_test, rf_predicted))


# # 55

# In[225]:


rows = ['LR','KNN','SVM','NB','DT','RF']
columns = ['ACC','F1 Score','Precision','Recall']

values = [[accuracy_score(y_test, logistic_predicted),f1_score(y_test, logistic_predicted),precision_score(y_test, logistic_predicted), recall_score(y_test, logistic_predicted)],
         [accuracy_score(y_test, knn_predicted),f1_score(y_test, knn_predicted),precision_score(y_test, knn_predicted), recall_score(y_test, knn_predicted)],
         [accuracy_score(y_test, svm_predicted),f1_score(y_test, svm_predicted),precision_score(y_test, svm_predicted), recall_score(y_test, svm_predicted)],
         [accuracy_score(y_test, nb_predicted),f1_score(y_test, nb_predicted),precision_score(y_test, nb_predicted), recall_score(y_test, nb_predicted)],
         [accuracy_score(y_test, dt_predicted),f1_score(y_test, dt_predicted),precision_score(y_test, dt_predicted), recall_score(y_test, dt_predicted)],
         [accuracy_score(y_test, rf_predicted),f1_score(y_test, rf_predicted),precision_score(y_test, rf_predicted), recall_score(y_test, rf_predicted)]]


# In[226]:


#เนื่องจากขึ้น error ไม่แสดงผล จาก SVM เราก็เลยลบทิ้ง


# In[227]:


rows = ['LR','KNN','NB','DT','RF']
columns = ['ACC','F1 Score','Precision','Recall']

values = [[accuracy_score(y_test, logistic_predicted),f1_score(y_test, logistic_predicted),precision_score(y_test, logistic_predicted), recall_score(y_test, logistic_predicted)],
         [accuracy_score(y_test, knn_predicted),f1_score(y_test, knn_predicted),precision_score(y_test, knn_predicted), recall_score(y_test, knn_predicted)],
         [accuracy_score(y_test, nb_predicted),f1_score(y_test, nb_predicted),precision_score(y_test, nb_predicted), recall_score(y_test, nb_predicted)],
         [accuracy_score(y_test, dt_predicted),f1_score(y_test, dt_predicted),precision_score(y_test, dt_predicted), recall_score(y_test, dt_predicted)],
         [accuracy_score(y_test, rf_predicted),f1_score(y_test, rf_predicted),precision_score(y_test, rf_predicted), recall_score(y_test, rf_predicted)]]


# In[228]:


df_model_compare = pd.DataFrame(values,rows,columns)

df_model_compare


# In[229]:


fig = px.bar(df_model_compare, y='ACC', x=df_model_compare.index, title='ACC Comparison')
fig.show()


# In[230]:


fig = px.bar(df_model_compare, y='F1 Score', x=df_model_compare.index, title='F1 Score Comparison')
fig.show()


# In[231]:


fig = px.bar(df_model_compare, y='Precision', x=df_model_compare.index, title='Precision Comparison')
fig.show()


# In[232]:


fig = px.bar(df_model_compare, y='Recall', x=df_model_compare.index, title='Recall')
fig.show()


# # 56  Support Vector Machine + Normalized

# In[ ]:


# SVM ไม่ใช้ standardization เพราะว่า เป็นการแจกแจกไม่ปกติ


# In[233]:


from sklearn.preprocessing import MinMaxScaler


# In[234]:


min_max_scaler = MinMaxScaler()


# In[235]:


df_real


# In[236]:


df_minmax = min_max_scaler.fit_transform(df_real)


# In[237]:


df_minmax


# In[238]:


df_norm = pd.DataFrame(df_minmax, columns=df_real.columns)

df_norm


# In[239]:


X_norm = df_norm.drop(['Churn'], axis=1)
X_norm


# In[240]:


y_norm = df_norm['Churn']
y_norm


# In[241]:


X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(X_norm, y_norm, test_size=0.2, random_state=100)


# In[242]:


svc_norm = SVC()
svc_norm.fit(X_train, y_train)


# In[243]:


svm_norm_predicted = svc_norm.predict(X_test)


# # 57

# In[244]:


fig = plt.figure(figsize=(12,8))
sns.countplot(svm_norm_predicted)


# # 58

# In[245]:


confusion_matrix(y_test, svm_norm_predicted)


# In[246]:


print('Support Vector Machine + Normalized Result')

print('Accuracy: ', accuracy_score(y_test, svm_norm_predicted))
print('F1 Score: ', f1_score(y_test, svm_norm_predicted))
print('Precision: ', precision_score(y_test, svm_norm_predicted))
print('Recall: ', recall_score(y_test, svm_norm_predicted))


# # 59 Support Vector Machine + Normalized + GridSearch

# In[247]:


from sklearn.model_selection import GridSearchCV


# In[249]:


param_combination = {'C':[0.01,0.1,1,10,100,1000,10000], 'gamma':[0.00001,0.0001,0.001,0.01,0.1,1,10]}


# In[250]:


grid_search =  GridSearchCV(SVC(), param_combination, verbose=3)


# In[251]:


grid_search.fit(X_train_norm, y_train_norm)


# In[252]:


grid_search.best_params_


# In[253]:


grid_search.best_estimator_


# In[254]:


svm_norm_grid_predicted = grid_search.predict(X_test_norm)

svm_norm_grid_predicted


# # 60

# In[255]:


fig = plt.figure(figsize=(12,8))
sns.countplot(svm_norm_grid_predicted)


# # 61

# In[256]:


confusion_matrix(y_test, svm_norm_grid_predicted)


# In[257]:


print('Support Vector Machine + Normalized + GridSearch Result')

print('Accuracy: ', accuracy_score(y_test, svm_norm_grid_predicted))
print('F1 Score: ', f1_score(y_test, svm_norm_grid_predicted))
print('Precision: ', precision_score(y_test, svm_norm_grid_predicted))
print('Recall: ', recall_score(y_test, svm_norm_grid_predicted))


# # 62 Random Forest + GridSearch  (Hyperparameter tuning)

# In[263]:


param_grid = {'max_depth':[4,8,16,None],
              'max_features':[4,8],
              'n_estimators':[50,100,200,500],
              'min_samples_split':[3,5,6,7]}


# In[264]:


grid_search =  GridSearchCV(RandomForestClassifier(n_jobs=-1), param_grid, verbose=3)

# n_jobs=-1 คือการสั่งให้คอมพิวเตอร์ใช้ทุก core ที่มีอยู่มาช่วยๆกันสร้างต้นไม้


# In[265]:


grid_search.fit(X_train, y_train)


# In[266]:


grid_search.best_params_


# In[267]:


grid_search.best_estimator_


# In[274]:


rf_grid_predicted = grid_search.predict(X_test)

rf_grid_predicted


# # 63

# In[276]:


fig = plt.figure(figsize=(12,8))
sns.countplot(rf_grid_predicted)


# # 64

# In[278]:


confusion_matrix(y_test, rf_grid_predicted)


# In[279]:


print('Random Forest + GridSearch Result')

print('Accuracy: ', accuracy_score(y_test, rf_grid_predicted))
print('F1 Score: ', f1_score(y_test, rf_grid_predicted))
print('Precision: ', precision_score(y_test, rf_grid_predicted))
print('Recall: ', recall_score(y_test, rf_grid_predicted))


# # 65 - 68

# In[281]:


rows = ['LR','KNN','NB','DT','RF','SVM with HT','RF with HT']
columns = ['ACC','F1 Score','Precision','Recall']

values = [[accuracy_score(y_test, logistic_predicted),f1_score(y_test, logistic_predicted),precision_score(y_test, logistic_predicted), recall_score(y_test, logistic_predicted)],
         [accuracy_score(y_test, knn_predicted),f1_score(y_test, knn_predicted),precision_score(y_test, knn_predicted), recall_score(y_test, knn_predicted)],
         [accuracy_score(y_test, nb_predicted),f1_score(y_test, nb_predicted),precision_score(y_test, nb_predicted), recall_score(y_test, nb_predicted)],
         [accuracy_score(y_test, dt_predicted),f1_score(y_test, dt_predicted),precision_score(y_test, dt_predicted), recall_score(y_test, dt_predicted)],
         [accuracy_score(y_test, rf_predicted),f1_score(y_test, rf_predicted),precision_score(y_test, rf_predicted), recall_score(y_test, rf_predicted)],
         [accuracy_score(y_test, svm_norm_grid_predicted),f1_score(y_test, svm_norm_grid_predicted),precision_score(y_test, svm_norm_grid_predicted), recall_score(y_test, svm_norm_grid_predicted)],
         [accuracy_score(y_test, rf_grid_predicted),f1_score(y_test, rf_grid_predicted),precision_score(y_test, rf_grid_predicted), recall_score(y_test, rf_grid_predicted)]]


# In[282]:


df_model_full_compare = pd.DataFrame(values,rows,columns)

df_model_full_compare


# In[283]:


fig = px.bar(df_model_full_compare, y='ACC', x=df_model_full_compare.index, title='ACC Comparison')
fig.show()


# In[284]:


fig = px.bar(df_model_full_compare, y='F1 Score', x=df_model_full_compare.index, title='F1 Score Comparison')
fig.show()


# In[285]:


fig = px.bar(df_model_full_compare, y='Precision', x=df_model_full_compare.index, title='Precision Comparison')
fig.show()


# In[286]:


fig = px.bar(df_model_full_compare, y='Recall', x=df_model_full_compare.index, title='Recall Comparison')
fig.show()


# In[ ]:




