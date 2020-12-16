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


df = pd.read_csv('data_student.csv')
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

# In[10]:


df.corr()


# # 6

# In[11]:


plt.title('Best correlation')
plt.xlabel('STG')
plt.ylabel('PEG')
plt.scatter(df['STG'], df['PEG'])


# In[12]:


plt.title('Worst correlation')
plt.xlabel('LPR')
plt.ylabel('PEG')
plt.scatter(df['LPR'], df['PEG'])


# # 7

# In[15]:


sns.distplot(df['STG'])


# In[16]:


sns.distplot(df['SCG'])


# In[17]:


sns.distplot(df['STR'])


# In[18]:


sns.distplot(df['LPR'])


# In[19]:


sns.distplot(df['PEG'])


# # 8

# In[25]:


sns.boxplot(x=' UNS',y='STG', data=df)


# In[26]:


sns.boxplot(x=' UNS',y='SCG', data=df)


# In[27]:


sns.boxplot(x=' UNS',y='STR', data=df)


# In[28]:


sns.boxplot(x=' UNS',y='LPR', data=df)


# In[29]:


sns.boxplot(x=' UNS',y='PEG', data=df)


# # 9 K-means

# In[49]:


from sklearn.cluster import KMeans


# In[65]:


df[['STG','PEG']]


# In[66]:


X = df[['STG','PEG']].values

X


# In[67]:


kmeans = KMeans(n_clusters = 2, random_state = 99)


# In[68]:


kmeans_label = kmeans.fit_predict(X)
kmeans_label 


# # 10

# In[69]:


kmeans.cluster_centers_


# In[70]:


kmeans.inertia_


# In[71]:


kmeans.n_iter_


# # 11

# In[72]:


#เลือกค่า x1,x2,y1,y2 เพื่อทำ clustering virtualization


# In[74]:


x1 = X[kmeans_label == 0][:,0]
y1 = X[kmeans_label == 0][:,1]
x2 = X[kmeans_label == 1][:,0]
y2 = X[kmeans_label == 1][:,1]


# In[75]:


x1


# In[79]:


fig = plt.figure(figsize=(12,8))
plt.scatter(x1, y1, s=20, c='blue', label='Cluster 1')
plt.scatter(x2, y2, s=20, c='red', label='Cluster 2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='green', label='Centroids')
plt.title('Cluster of Student with K=2')
plt.xlabel('STG')
plt.ylabel('PEG')
plt.legend()
plt.grid()
plt.show()


# # 12 Elbow Method

# In[82]:


wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 99)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
fig = plt.figure(figsize=(12,8))
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()


# In[83]:


wcss


# # 13  Kmeans = 4

# In[84]:


kmeans = KMeans(n_clusters = 4, random_state = 99)


# In[85]:


kmeans_label = kmeans.fit_predict(X)
kmeans_label 


# In[86]:


kmeans.cluster_centers_


# In[87]:


kmeans.inertia_


# In[88]:


kmeans.n_iter_


# # 14

# In[89]:


x1 = X[kmeans_label == 0][:,0]
y1 = X[kmeans_label == 0][:,1]
x2 = X[kmeans_label == 1][:,0]
y2 = X[kmeans_label == 1][:,1]
x3 = X[kmeans_label == 2][:,0]
y3 = X[kmeans_label == 2][:,1]
x4 = X[kmeans_label == 3][:,0]
y4 = X[kmeans_label == 3][:,1]


# In[90]:


x1


# In[91]:


fig = plt.figure(figsize=(12,8))
plt.scatter(x1, y1, s=20, c='blue', label='Cluster 1')
plt.scatter(x2, y2, s=20, c='red', label='Cluster 2')
plt.scatter(x3, y3, s=20, c='yellow', label='Cluster 3')
plt.scatter(x4, y4, s=20, c='pink', label='Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='green', label='Centroids')
plt.title('Cluster of Student with K=4')
plt.xlabel('STG')
plt.ylabel('PEG')
plt.legend()
plt.grid()
plt.show()


# # 15  เลือก  features ใหม่  และ k = 2

# In[105]:


df[['LPR','STR']]


# In[106]:


X = df[['LPR','STR']].values

X


# In[107]:


kmeans = KMeans(n_clusters = 2, random_state = 99)

kmeans_label = kmeans.fit_predict(X)
kmeans_label 


# # 16

# In[108]:


kmeans.cluster_centers_


# # 17

# In[109]:


kmeans.inertia_


# In[110]:


kmeans.n_iter_


# # 18

# In[111]:


x1 = X[kmeans_label == 0][:,0]
y1 = X[kmeans_label == 0][:,1]
x2 = X[kmeans_label == 1][:,0]
y2 = X[kmeans_label == 1][:,1]


# In[112]:


x1


# In[113]:


fig = plt.figure(figsize=(12,8))
plt.scatter(x1, y1, s=20, c='blue', label='Cluster 1')
plt.scatter(x2, y2, s=20, c='red', label='Cluster 2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='green', label='Centroids')
plt.title('Cluster of Student with K=2')
plt.xlabel('LPR')
plt.ylabel('STR')
plt.legend()
plt.grid()
plt.show()


# # 19

# In[114]:


wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 99)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
fig = plt.figure(figsize=(12,8))
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()


# In[115]:


wcss


# # 20  เลือก k ที่น่าจะดีที่สุด คือ k = 4

# In[119]:


kmeans = KMeans(n_clusters = 4, random_state = 99)

kmeans_label = kmeans.fit_predict(X)
kmeans_label 


# In[120]:


kmeans.cluster_centers_


# In[121]:


kmeans.inertia_


# In[122]:


kmeans.n_iter_


# # 21

# In[123]:


x1 = X[kmeans_label == 0][:,0]
y1 = X[kmeans_label == 0][:,1]
x2 = X[kmeans_label == 1][:,0]
y2 = X[kmeans_label == 1][:,1]
x3 = X[kmeans_label == 2][:,0]
y3 = X[kmeans_label == 2][:,1]
x4 = X[kmeans_label == 3][:,0]
y4 = X[kmeans_label == 3][:,1]


# In[124]:


x1


# In[125]:


fig = plt.figure(figsize=(12,8))
plt.scatter(x1, y1, s=20, c='blue', label='Cluster 1')
plt.scatter(x2, y2, s=20, c='red', label='Cluster 2')
plt.scatter(x3, y3, s=20, c='yellow', label='Cluster 3')
plt.scatter(x4, y4, s=20, c='pink', label='Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='green', label='Centroids')
plt.title('Cluster of Student with K=4')
plt.xlabel('LPR')
plt.ylabel('STR')
plt.legend()
plt.grid()
plt.show()


# In[ ]:




