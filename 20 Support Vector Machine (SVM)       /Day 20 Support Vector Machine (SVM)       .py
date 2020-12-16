#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[65]:


#1


# In[66]:


df = pd.read_csv('Prostate_Cancer.csv')
df


# In[67]:


#2


# In[68]:


df.head(10)


# In[69]:


df.tail(10)


# In[70]:


df.sample(10)


# In[71]:


#3 + 4


# In[72]:


df.info()


# In[73]:


df.describe()


# In[74]:


#5


# In[75]:


df = df.drop('id', axis=1)

df


# In[76]:


sns.pairplot(df)


# In[77]:


#6


# In[78]:


sns.distplot(df['radius'])


# In[79]:


sns.distplot(df['texture'])


# In[80]:


sns.distplot(df['perimeter'])


# In[81]:


sns.distplot(df['area'])


# In[82]:


sns.distplot(df['smoothness'])


# In[83]:


sns.distplot(df['compactness'])


# In[84]:


sns.distplot(df['symmetry'])


# In[85]:


sns.distplot(df['fractal_dimension'])


# In[86]:


#7


# In[87]:


df.corr()


# In[88]:


sns.heatmap(df.corr())


# In[ ]:


#8


# In[33]:


plt.title('Best correlation')
plt.xlabel('area')
plt.ylabel('perimeter')
plt.scatter(df['area'], df['perimeter'])


# In[34]:


#9


# In[35]:


fig = plt.figure(figsize=(12,8))
plt.title('Worst correlation')
plt.xlabel('area')
plt.ylabel('fractal_dimension')
plt.scatter(df['area'], df['fractal_dimension'])


# In[36]:


#10


# In[89]:


fig = plt.figure(figsize=(12,8))
plt.hist(df['area'],bins=10)
plt.show()


# In[90]:


#11


# In[91]:


fig = plt.figure(figsize=(12,8))
sns.boxplot(x='diagnosis_result',y='perimeter', data=df)


# In[92]:


#12


# In[93]:


from sklearn.model_selection import train_test_split


# In[94]:


df


# In[95]:


result = pd.get_dummies(df['diagnosis_result'])

result


# In[96]:


result = pd.get_dummies(df['diagnosis_result'], drop_first=True)

result


# In[97]:


df = pd.concat([df,result],axis=1)
df


# In[98]:


df = df.drop('diagnosis_result', axis=1)

df


# In[108]:


df['result name'] = ['M' if x==1 else 'B' for x in df['M']]

df['result name']


# In[109]:


df


# In[113]:


X = df.drop(['M', 'result name'], axis = 1)
X


# In[114]:


y = df['M']
y


# In[115]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[116]:


X_train


# In[47]:


#13


# In[117]:


sns.distplot(df['perimeter'])


# In[118]:


#14


# ควรทำ Normalization เนื่องจากข้อมูลมีค่าที่หลากหลาย และข้อมูลมีการกระจายตัวที่ไม่ปกติ

# In[ ]:


#15 แบบ default


# In[119]:


from sklearn.svm import SVC


# In[120]:


svc = SVC()


# In[121]:


svc.fit(X_train, y_train)


# In[122]:


predicted = svc.predict(X_test)
predicted


# In[123]:


len(predicted)


# In[124]:


#15  แบบทำ normalize


# In[125]:


from sklearn.preprocessing import MinMaxScaler


# In[126]:


min_max_scaler = MinMaxScaler()


# In[128]:


arr_minmax = min_max_scaler.fit_transform(df.drop(['M', 'result name'], axis = 1))
arr_minmax


# In[132]:


new_df = pd.DataFrame(arr_minmax, columns = df.columns[:-2])

new_df


# In[134]:


X2 = new_df
X2


# In[135]:


y2 = df['M']
y2


# In[136]:


X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=100)


# In[137]:


svc.fit(X2_train, y2_train)


# In[138]:


predicted2 = svc.predict(X2_test)
predicted2


# In[139]:


#16  วัดผล


# In[140]:


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


# In[141]:


#วัดผล ของ default ก่อน


# In[142]:


confusion_matrix(y_test,predicted)


# In[143]:


print('Accuracy = ', accuracy_score(y_test,predicted))
print('F1-score = ', f1_score(y_test,predicted))
print('Precision = ', precision_score(y_test,predicted))
print('Recall = ', recall_score(y_test,predicted))


# In[144]:


#วัดผล ของ Normalization


# In[145]:


confusion_matrix(y2_test,predicted2)


# In[146]:


print('Accuracy = ', accuracy_score(y2_test,predicted2))
print('F1-score = ', f1_score(y2_test,predicted2))
print('Precision = ', precision_score(y2_test,predicted2))
print('Recall = ', recall_score(y2_test,predicted2))


# In[147]:


#17  หาค่า parameter combination ที่ดีที่สุด


# In[150]:


from sklearn.model_selection import GridSearchCV


# In[151]:


param_combination = {'C':[0.01,0.1,1,10,100,1000,10000], 'gamma':[0.00001,0.0001,0.001,0.01,0.1,1,10]}


# In[152]:


grid_search = GridSearchCV(SVC(), param_combination, verbose=3)


# In[153]:


grid_search.fit(X_train, y_train)


# In[154]:


grid_search.best_params_


# In[155]:


grid_search.best_estimator_


# In[156]:


grid_predicted = grid_search.predict(X_test)
grid_predicted


# In[157]:


confusion_matrix(y_test,grid_predicted)


# In[158]:


print('Accuracy = ', accuracy_score(y_test,grid_predicted))
print('F1-score = ', f1_score(y_test,grid_predicted))
print('Precision = ', precision_score(y_test,grid_predicted))
print('Recall = ', recall_score(y_test,grid_predicted))


# In[159]:


#18 เลือกบาง feature บางเทรนโมเดล เปรียบเทียบกับ all feature


# In[160]:


df


# In[166]:


X


# In[167]:


X.corr()


# In[168]:


sns.heatmap(X.corr())


# In[169]:


X3 = df[['area','perimeter','symmetry','compactness']]
X3


# In[170]:


y3 = df['M']
y3


# In[171]:


X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=100)


# In[172]:


X3_train


# In[173]:


svc.fit(X3_train, y3_train)


# In[174]:


predicted3 = svc.predict(X3_test)
predicted3


# In[175]:


confusion_matrix(y3_test,predicted3)


# In[177]:


#วัดผลของ ที่เลือกบาง feature

print('Accuracy = ', accuracy_score(y3_test,predicted3))
print('F1-score = ', f1_score(y3_test,predicted3))
print('Precision = ', precision_score(y3_test,predicted3))
print('Recall = ', recall_score(y3_test,predicted3))


# In[178]:


#วัดผลของ ทุก feature


# In[179]:


print('Accuracy = ', accuracy_score(y_test,predicted))
print('F1-score = ', f1_score(y_test,predicted))
print('Precision = ', precision_score(y_test,predicted))
print('Recall = ', recall_score(y_test,predicted))


# In[180]:


#19 - 21


# In[201]:


Score = pd.DataFrame({
    'Type': ['Default','Grid Search','Normalization'],
    'F1 Score' : [f1_score(y_test,predicted),f1_score(y_test,grid_predicted),f1_score(y2_test,predicted2)],
    'Recall': [recall_score(y_test,predicted),recall_score(y_test,grid_predicted),recall_score(y2_test,predicted2)],
    'Accuracy': [accuracy_score(y_test,predicted),accuracy_score(y_test,grid_predicted),accuracy_score(y2_test,predicted2)]})


# In[202]:


Score


# In[205]:


Score.plot(kind='bar')
plt.show()


# In[206]:


Score.set_index('Type')


# In[209]:


Score = Score.set_index('Type')


# In[210]:


Score.plot(kind='bar')
plt.show()


# In[211]:


#22   เอาทุกอย่างมายำรวมกัน  Default + Normalization + Grid Search


# In[213]:


# Normalized แล้ว

new_df


# In[214]:


X2


# In[215]:


y2


# In[216]:


grid_search.fit(X2_train, y2_train)


# In[217]:


grid_search.best_params_


# In[218]:


grid_search.best_estimator_


# In[219]:


grid_predicted2 = grid_search.predict(X2_test)
grid_predicted2


# In[220]:


confusion_matrix(y2_test,grid_predicted2)


# In[222]:


# แบบเอาทุกอย่างมายำรวมกัน Default + Normalization + Grid Search

print('Accuracy = ', accuracy_score(y2_test,grid_predicted2))
print('F1-score = ', f1_score(y2_test,grid_predicted2))
print('Precision = ', precision_score(y2_test,grid_predicted2))
print('Recall = ', recall_score(y2_test,grid_predicted2))


# In[223]:


# แบบ Default

print('Accuracy = ', accuracy_score(y_test,predicted))
print('F1-score = ', f1_score(y_test,predicted))
print('Precision = ', precision_score(y_test,predicted))
print('Recall = ', recall_score(y_test,predicted))


# In[224]:


# แบบ Grid search อย่างเดียว

print('Accuracy = ', accuracy_score(y_test,grid_predicted))
print('F1-score = ', f1_score(y_test,grid_predicted))
print('Precision = ', precision_score(y_test,grid_predicted))
print('Recall = ', recall_score(y_test,grid_predicted))


# In[225]:


# แบบ Normalized อย่างเดียว

print('Accuracy = ', accuracy_score(y2_test,predicted2))
print('F1-score = ', f1_score(y2_test,predicted2))
print('Precision = ', precision_score(y2_test,predicted2))
print('Recall = ', recall_score(y2_test,predicted2))


# In[ ]:




