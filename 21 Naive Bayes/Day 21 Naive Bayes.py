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


# In[41]:


df = pd.read_csv('glass.csv')
df


# In[42]:


#2


# In[43]:


df.head(10)


# In[44]:


df.tail(10)


# In[45]:


df.sample(10)


# In[46]:


#3 + 4


# In[47]:


df.info()


# In[48]:


df.describe()


# In[49]:


#5


# In[50]:


df_drop = df.drop('Type', axis=1)

df_drop


# In[51]:


sns.pairplot(df_drop)


# In[29]:


#6


# In[52]:


sns.distplot(df['RI'])


# In[53]:


sns.distplot(df['Na'])


# In[54]:


#7


# In[55]:


df_drop.corr()


# In[56]:


sns.heatmap(df_drop.corr())


# In[34]:


#8


# In[57]:


plt.title('Best correlation')
plt.xlabel('RI')
plt.ylabel('Ca')
plt.scatter(df['RI'], df['Ca'])


# In[58]:


#9


# In[59]:


fig = plt.figure(figsize=(12,8))
plt.title('Worst correlation')
plt.xlabel('RI')
plt.ylabel('Si')
plt.scatter(df['RI'], df['Si'])


# In[60]:


#10


# In[61]:


fig = plt.figure(figsize=(12,8))
plt.hist(df['Mg'],bins=10)
plt.show()


# In[40]:


#11


# In[62]:


fig = plt.figure(figsize=(12,8))
sns.boxplot(x='Type',y='RI', data=df)


# In[63]:


#12


# In[202]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_drop, df['Type'], test_size=0.2, random_state=100)


# In[203]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[204]:


#13


# In[205]:


sns.distplot(df['RI'])


# In[206]:


#14


# ควรทำ Normalization เนื่องจากข้อมูลมีค่าที่หลากหลาย และข้อมูลมีการกระจายตัวที่ไม่ปกติ

# In[207]:


#15


# In[208]:


from sklearn.naive_bayes import GaussianNB

#default all features

nb = GaussianNB()
nb.fit(X_train, y_train)


# In[209]:


predicted = nb.predict(X_test)
predicted


# In[210]:


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


# In[211]:


confusion_matrix(y_test,predicted)


# In[212]:


#default all features

print('Accuracy = ', accuracy_score(y_test,predicted))
print('F1-score = ', f1_score(y_test,predicted, average='micro'))
print('Precision = ', precision_score(y_test,predicted, average='micro'))
print('Recall = ', recall_score(y_test,predicted, average='micro'))


# In[213]:


#normalized all features   #ิbefore split

df_drop


# In[214]:


from sklearn.preprocessing import MinMaxScaler


# In[215]:


min_max_scaler = MinMaxScaler()


# In[216]:


arr_minmax = min_max_scaler.fit_transform(df_drop)

arr_minmax


# In[217]:


norm_df = pd.DataFrame(arr_minmax, columns=df_drop.columns)

norm_df


# In[218]:


X_train, X_test, y_train, y_test = train_test_split(norm_df, df['Type'], test_size=0.2, random_state=100)


# In[219]:


#normalized all features   #ิbefore split

nb2 = GaussianNB()
nb2.fit(X_train, y_train)


# In[220]:


predicted2 = nb2.predict(X_test)
predicted2


# In[221]:


confusion_matrix(y_test,predicted2)


# In[222]:


#normalized all features   #ิbefore split

print('Accuracy = ', accuracy_score(y_test,predicted2))
print('F1-score = ', f1_score(y_test,predicted2, average='micro'))
print('Precision = ', precision_score(y_test,predicted2, average='micro'))
print('Recall = ', recall_score(y_test,predicted2, average='micro'))


# In[223]:


#16  เลือกบาง feature มาเทรนโมเดล


# In[224]:


X = df[['RI','Ca']]
X


# In[225]:


X_train, X_test, y_train, y_test = train_test_split(X, df['Type'], test_size=0.2, random_state=100)


# In[226]:


X_train


# In[227]:


#default some features

nb3 = GaussianNB()
nb3.fit(X_train, y_train)


# In[228]:


predicted3 = nb3.predict(X_test)
predicted3


# In[229]:


confusion_matrix(y_test,predicted3)


# In[230]:


#default some features

print('Accuracy = ', accuracy_score(y_test,predicted3))
print('F1-score = ', f1_score(y_test,predicted3, average='micro'))
print('Precision = ', precision_score(y_test,predicted3, average='micro'))
print('Recall = ', recall_score(y_test,predicted3, average='micro'))


# In[231]:


#17 


# In[232]:


#standardized all features   #ิbefore split


# In[233]:


df_drop


# In[234]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[235]:


arr_scale = sc.fit_transform(df_drop)

arr_scale


# In[236]:


sc_df = pd.DataFrame(arr_scale, columns=df_drop.columns)

sc_df


# In[237]:


X_train, X_test, y_train, y_test = train_test_split(sc_df, df['Type'], test_size=0.2, random_state=100)


# In[238]:


X_train


# In[239]:


#standardized all features   #ิbefore split

nb4 = GaussianNB()
nb4.fit(X_train, y_train)


# In[240]:


predicted4 = nb4.predict(X_test)
predicted4


# In[241]:


confusion_matrix(y_test,predicted4)


# In[242]:


#standardized all features   #ิbefore split

print('Accuracy = ', accuracy_score(y_test,predicted4))
print('F1-score = ', f1_score(y_test,predicted4, average='micro'))
print('Precision = ', precision_score(y_test,predicted4, average='micro'))
print('Recall = ', recall_score(y_test,predicted4, average='micro'))


# In[243]:


#ต่อไปจะทำ Normalize กับ Standardize แบบหลังแบ่งข้อมูล


# In[244]:


X_train, X_test, y_train, y_test = train_test_split(df_drop, df['Type'], test_size=0.2, random_state=100)


# In[245]:


X_train


# In[246]:


#Normalization after split


# In[247]:


min_max_scaler2 = MinMaxScaler()


# In[248]:


X_train = min_max_scaler2.fit_transform(X_train)

X_train


# In[249]:


X_train = pd.DataFrame(X_train, columns=df_drop.columns)

X_train


# In[250]:


min_max_scaler3 = MinMaxScaler()


# In[251]:


X_test = min_max_scaler3.fit_transform(X_test)

X_test


# In[252]:


X_test = pd.DataFrame(X_test, columns=df_drop.columns)

X_test


# In[253]:


#Normalized after split

nb5 = GaussianNB()
nb5.fit(X_train, y_train)


# In[254]:


predicted5 = nb5.predict(X_test)
predicted5


# In[255]:


confusion_matrix(y_test,predicted5)


# In[256]:


#Normalized after split

print('Accuracy = ', accuracy_score(y_test,predicted5))
print('F1-score = ', f1_score(y_test,predicted5, average='micro'))
print('Precision = ', precision_score(y_test,predicted5, average='micro'))
print('Recall = ', recall_score(y_test,predicted5, average='micro'))


# In[257]:


#Standardization after split


# In[333]:


X_train, X_test, y_train, y_test = train_test_split(df_drop, df['Type'], test_size=0.2, random_state=100)


# In[334]:


sc2 = StandardScaler()
sc3 = StandardScaler()


# In[335]:


X_train = sc2.fit_transform(X_train)

X_train


# In[336]:


X_train = pd.DataFrame(X_train, columns=df_drop.columns)

X_train


# In[337]:


X_test = sc3.fit_transform(X_test)

X_test


# In[338]:


X_test = pd.DataFrame(X_test, columns=df_drop.columns)

X_test


# In[264]:


#Standardized after split

nb6 = GaussianNB()
nb6.fit(X_train, y_train)


# In[265]:


predicted6 = nb6.predict(X_test)
predicted6


# In[266]:


confusion_matrix(y_test,predicted6)


# In[267]:


#Standardized after split

print('Accuracy = ', accuracy_score(y_test,predicted6))
print('F1-score = ', f1_score(y_test,predicted6, average='micro'))
print('Precision = ', precision_score(y_test,predicted6, average='micro'))
print('Recall = ', recall_score(y_test,predicted6, average='micro'))


# In[268]:


#visualize


# In[272]:


Score_before_split = pd.DataFrame({
    'Type': ['Normalization','Standardization'],
    'F1 Score' : [f1_score(y_test,predicted2, average='micro'),f1_score(y_test,predicted4, average='micro')]})


# In[273]:


Score_before_split


# In[274]:


Score_before_split.set_index('Type')


# In[275]:


Score_before_split = Score_before_split.set_index('Type')


# In[276]:


#Score_before_split

Score_before_split.plot(kind='bar')
plt.show()


# In[277]:


Score_after_split = pd.DataFrame({
    'Type': ['Normalization','Standardization'],
    'F1 Score' : [f1_score(y_test,predicted5, average='micro'),f1_score(y_test,predicted6, average='micro')]})


# In[278]:


Score_after_split = Score_after_split.set_index('Type')

Score_after_split


# In[279]:


#Score_after_split

Score_after_split.plot(kind='bar')
plt.show()


# In[280]:


#18


# In[451]:


#ปรับพาริมิเตอร์

from sklearn.naive_bayes import GaussianNB

nb6 = GaussianNB(var_smoothing=1e-05)

nb6.fit(X_train, y_train)


# In[452]:


predicted6 = nb6.predict(X_test)
predicted6


# In[453]:


confusion_matrix(y_test,predicted6)


# In[454]:


#Standardized after split

print('Accuracy = ', accuracy_score(y_test,predicted6))
print('F1-score = ', f1_score(y_test,predicted6, average='micro'))
print('Precision = ', precision_score(y_test,predicted6, average='micro'))
print('Recall = ', recall_score(y_test,predicted6, average='micro'))


# #ลองปรับแล้วก็ยังได้ค่าเท่าเดิม
