#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # 1

# In[4]:


df = pd.read_csv('data_student.csv')
df


# # 2

# In[5]:


df.head(10)


# In[6]:


df.tail(10)


# In[7]:


df.sample(10)


# # 3

# In[8]:


df.info()


# In[9]:


df.describe()


# # 4

# In[10]:


sns.pairplot(df)


# # 5

# In[11]:


df.corr()


# # 6

# In[12]:


plt.title('Best correlation')
plt.xlabel('STG')
plt.ylabel('PEG')
plt.scatter(df['STG'], df['PEG'])


# In[13]:


plt.title('Worst correlation')
plt.xlabel('LPR')
plt.ylabel('PEG')
plt.scatter(df['LPR'], df['PEG'])


# # 7

# In[14]:


sns.distplot(df['STG'])


# In[15]:


sns.distplot(df['SCG'])


# In[16]:


sns.distplot(df['STR'])


# In[17]:


sns.distplot(df['LPR'])


# In[18]:


sns.distplot(df['PEG'])


# # 8

# In[19]:


sns.boxplot(x=' UNS',y='STG', data=df)


# In[20]:


sns.boxplot(x=' UNS',y='SCG', data=df)


# In[21]:


sns.boxplot(x=' UNS',y='STR', data=df)


# In[22]:


sns.boxplot(x=' UNS',y='LPR', data=df)


# In[23]:


sns.boxplot(x=' UNS',y='PEG', data=df)


# # 9 สร้าง Dendrogram เพื่อหาจำนวน cluster ที่เหมาะสม

# In[24]:


import scipy.cluster.hierarchy as sch


# In[26]:


df[['STG','PEG']]


# In[27]:


X = df[['STG','PEG']].values

X


# In[28]:


fig = plt.figure(figsize=(12,8))

dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

plt.title('Dendrogram')
plt.xlabel('Students')
plt.ylabel('Euclidean distances')


# # 10 Agglomerative Hierarchical Clustering

# In[29]:


from sklearn.cluster import AgglomerativeClustering


# In[30]:


hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)


# In[31]:


y_hc


# # 11

# In[33]:


#เลือกค่า x1,x2,y1,y2 เพื่อทำ clustering virtualization


# In[36]:


y_hc == 0


# In[37]:


X[y_hc == 0]


# In[38]:


x1 = X[y_hc == 0][:,0]
y1 = X[y_hc == 0][:,1]
x2 = X[y_hc == 1][:,0]
y2 = X[y_hc == 1][:,1]


# In[39]:


x1


# In[41]:


fig = plt.figure(figsize=(12,8))
plt.scatter(x1, y1, s=20, c='blue', label='Cluster 1')
plt.scatter(x2, y2, s=20, c='red', label='Cluster 2')
plt.title('Cluster of Student with cluster = 2')
plt.xlabel('STG')
plt.ylabel('PEG')
plt.legend()
plt.grid()
plt.show()


# # 12 เลือก feature ใหม่

# In[42]:


df[['LPR','STR']]


# In[43]:


X = df[['LPR','STR']].values

X


# # 13 Dendrogram

# In[44]:


fig = plt.figure(figsize=(12,8))

dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

plt.title('Dendrogram')
plt.xlabel('Students')
plt.ylabel('Euclidean distances')


# # 14 Agglomerative Hierarchical Clustering = 4

# In[45]:


hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)


# In[46]:


y_hc


# # 15 Clustering Visualization

# In[47]:


x1 = X[y_hc == 0][:,0]
y1 = X[y_hc == 0][:,1]
x2 = X[y_hc == 1][:,0]
y2 = X[y_hc == 1][:,1]
x3 = X[y_hc == 2][:,0]
y3 = X[y_hc == 2][:,1]
x4 = X[y_hc == 3][:,0]
y4 = X[y_hc == 3][:,1]


# In[48]:


x1


# In[50]:


fig = plt.figure(figsize=(12,8))
plt.scatter(x1, y1, s=20, c='blue', label='Cluster 1')
plt.scatter(x2, y2, s=20, c='red', label='Cluster 2')
plt.scatter(x3, y3, s=20, c='yellow', label='Cluster 3')
plt.scatter(x4, y4, s=20, c='pink', label='Cluster 4')
plt.title('Cluster of Student with cluster = 4')
plt.xlabel('LPR')
plt.ylabel('STR')
plt.legend()
plt.grid()
plt.show()


# In[ ]:




