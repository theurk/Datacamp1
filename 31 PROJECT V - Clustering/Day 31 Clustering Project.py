#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # 1

# In[2]:


df = pd.read_csv('College.csv')
df


# # 2

# In[3]:


df.head(10)


# In[4]:


df.tail(10)


# In[5]:


df.sample(10)


# # 3

# In[7]:


df.info()


# In[8]:


df.info


# In[10]:


df.describe()


# # 4

# In[11]:


sns.pairplot(df)


# # 5

# In[12]:


df.corr()


# # 6

# In[13]:


fig = plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linecolor='white', linewidth=2)


# # 7

# In[14]:


plt.title('Best correlation')
plt.xlabel('Enroll')
plt.ylabel('F.Undergrad')
plt.scatter(df['Enroll'], df['F.Undergrad'])


# # 8

# In[15]:


plt.title('Worst correlation')
plt.xlabel('S.F.Ratio')
plt.ylabel('Expend')
plt.scatter(df['S.F.Ratio'], df['Expend'])


# # 9

# In[16]:


plt.title('Nearest zero correlation')
plt.xlabel('Grad.Rate')
plt.ylabel('ฺBooks')
plt.scatter(df['Grad.Rate'], df['Books'])


# # 10

# In[17]:


fig, [[ax1, ax2, ax3],[ ax4, ax5, ax6],[ax7, ax8, ax9]] = plt.subplots(3, 3, figsize=[12,10])
      
sns.boxplot(df['Grad.Rate'], orient='v', ax=ax1)
sns.boxplot(df['Books'], orient='v', ax=ax2)
sns.boxplot(df['S.F.Ratio'], orient='v', ax=ax3)
sns.boxplot(df['Expend'], orient='v', ax=ax4)
sns.boxplot(df['F.Undergrad'], orient='v', ax=ax5)
sns.boxplot(df['Enroll'], orient='v', ax=ax6)
sns.boxplot(df['Top25perc'], orient='v', ax=ax7)
sns.boxplot(df['P.Undergrad'], orient='v', ax=ax8)
sns.boxplot(df['Accept'], orient='v', ax=ax9)

fig.tight_layout()


# # 11

# In[20]:


sns.countplot(df['Private'])


# In[24]:


sns.countplot(data=df, x='Private')


# In[25]:


sns.countplot(data=df, y='Private')


# # 12

# In[26]:


sns.barplot(data=df, x='Private', y='Grad.Rate')


# # 13

# In[28]:


df.groupby('Private').mean()


# # 14

# In[29]:


import plotly.express as px


# In[30]:


fig = px.pie(df, values='Apps', names='Private', title='Sum of Applications')

fig.show()


# # 15

# In[31]:


df['Private'].value_counts()


# In[32]:


212/565


# In[34]:


1117530/565

#จำนวนคนสมัครมหาลัยเอกชน ต่อ 1 มหหาลัย


# In[35]:


1214743/212

#จำนวนคนสมัครมหาลัยรัฐ ต่อ 1 มหาลัย


# # 16

# In[39]:


fig = px.box(df, x='Private', y='Books', points='all',
            labels={'Private':'มหาวิทยาลัยเอกชนหรือไม่?','Books':'หนังสือ'})
                    
fig.show()


# # 17

# In[40]:


fig = px.imshow(df.corr())

fig.show()


# # 18

# In[41]:


df = df[df['Apps'] < 40000]

df


# # 19

# In[42]:


df.drop(['Unnamed: 0','Private'], axis=1)


# In[43]:


df = df.drop(['Unnamed: 0','Private'], axis=1)

df


# In[45]:


from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()


# In[46]:


df_scaled = min_max_scaler.fit_transform(df)


# In[47]:


df_scaled


# In[48]:


df = pd.DataFrame(data=df_scaled, columns=df.columns)

df


# # 20

# In[49]:


from sklearn.cluster import KMeans


# In[51]:


data = df[['Apps', 'Accept']]


# In[52]:


data


# In[53]:


data = np.array(data)

data


# In[54]:


kmeans = KMeans(n_clusters = 2)


# In[55]:


kmeans_label = kmeans.fit_predict(data)

kmeans_label


# # 21

# In[56]:


kmeans.cluster_centers_


# In[57]:


kmeans.inertia_


# In[58]:


kmeans.n_iter_


# # 22

# In[60]:


x1 = data[kmeans_label == 0][:,0]
y1 = data[kmeans_label == 0][:,1]
x2 = data[kmeans_label == 1][:,0]
y2 = data[kmeans_label == 1][:,1]


# In[61]:


x1


# In[62]:


fig = plt.figure(figsize=(12,8))
plt.scatter(x1, y1, s=20, c='blue', label='Cluster 1')
plt.scatter(x2, y2, s=20, c='red', label='Cluster 2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='green', label='Centroids')
plt.title('KMeans Clustering Visualization with K=2')
plt.xlabel('Apps')
plt.ylabel('Accept')
plt.legend()
plt.grid()
plt.show()


# # 23

# In[64]:


wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 99)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
    
fig = plt.figure(figsize=(12,8))
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()


# In[65]:


wcss


# # 24  สร้างจาก k = 4

# In[66]:


kmeans = KMeans(n_clusters = 4, random_state = 99)


# In[68]:


kmeans_label = kmeans.fit_predict(data)
kmeans_label 


# # 25

# In[69]:


kmeans.cluster_centers_


# In[70]:


kmeans.inertia_


# In[71]:


kmeans.n_iter_


# # 26

# In[72]:


x1 = data[kmeans_label == 0][:,0]
y1 = data[kmeans_label == 0][:,1]
x2 = data[kmeans_label == 1][:,0]
y2 = data[kmeans_label == 1][:,1]
x3 = data[kmeans_label == 2][:,0]
y3 = data[kmeans_label == 2][:,1]
x4 = data[kmeans_label == 3][:,0]
y4 = data[kmeans_label == 3][:,1]


# In[73]:


x1


# In[74]:


fig = plt.figure(figsize=(12,8))
plt.scatter(x1, y1, s=20, c='blue', label='Cluster 1')
plt.scatter(x2, y2, s=20, c='red', label='Cluster 2')
plt.scatter(x3, y3, s=20, c='yellow', label='Cluster 3')
plt.scatter(x4, y4, s=20, c='pink', label='Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='green', label='Centroids')
plt.title('KMeans Clustering Visualization with K=4')
plt.xlabel('Apps')
plt.ylabel('Accept')
plt.legend()
plt.grid()
plt.show()


# # 27 เลือก feature Top25perc, Outstate มาสร้าง k = 2

# In[91]:


data = df[['Top25perc','Outstate']]

data = np.array(data)

data


# In[92]:


kmeans = KMeans(n_clusters = 2)


# In[93]:


kmeans_label = kmeans.fit_predict(data)

kmeans_label


# # 28

# In[94]:


kmeans.cluster_centers_


# In[95]:


kmeans.inertia_


# In[96]:


kmeans.n_iter_


# # 29

# In[97]:


x1 = data[kmeans_label == 0][:,0]
y1 = data[kmeans_label == 0][:,1]
x2 = data[kmeans_label == 1][:,0]
y2 = data[kmeans_label == 1][:,1]


# In[98]:


x1


# In[99]:


fig = plt.figure(figsize=(12,8))
plt.scatter(x1, y1, s=20, c='blue', label='Cluster 1')
plt.scatter(x2, y2, s=20, c='red', label='Cluster 2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='green', label='Centroids')
plt.title('KMeans Clustering Visualization with K=2')
plt.xlabel('Top25perc')
plt.ylabel('Outstate')
plt.legend()
plt.grid()
plt.show()


# # 30

# In[100]:


wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 99)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
    
fig = plt.figure(figsize=(12,8))
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()


# In[101]:


wcss


# # 31   เลือก k = 6

# In[102]:


kmeans = KMeans(n_clusters = 6)


# In[103]:


kmeans_label = kmeans.fit_predict(data)

kmeans_label


# # 32

# In[104]:


kmeans.cluster_centers_


# In[105]:


kmeans.inertia_


# In[106]:


kmeans.n_iter_


# # 33

# In[107]:


x1 = data[kmeans_label == 0][:,0]
y1 = data[kmeans_label == 0][:,1]
x2 = data[kmeans_label == 1][:,0]
y2 = data[kmeans_label == 1][:,1]
x3 = data[kmeans_label == 2][:,0]
y3 = data[kmeans_label == 2][:,1]
x4 = data[kmeans_label == 3][:,0]
y4 = data[kmeans_label == 3][:,1]
x5 = data[kmeans_label == 4][:,0]
y5 = data[kmeans_label == 4][:,1]
x6 = data[kmeans_label == 5][:,0]
y6 = data[kmeans_label == 5][:,1]


# In[108]:


fig = plt.figure(figsize=(12,8))
plt.scatter(x1, y1, s=20, c='blue', label='Cluster 1')
plt.scatter(x2, y2, s=20, c='red', label='Cluster 2')
plt.scatter(x3, y3, s=20, c='yellow', label='Cluster 3')
plt.scatter(x4, y4, s=20, c='pink', label='Cluster 4')
plt.scatter(x5, y5, s=20, c='orange', label='Cluster 5')
plt.scatter(x6, y6, s=20, c='purple', label='Cluster 6')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='green', label='Centroids')
plt.title('KMeans Clustering Visualization with K=6')
plt.xlabel('Top25perc')
plt.ylabel('Outstate')
plt.legend()
plt.grid()
plt.show()


# # 34  Hierarchy Clustering

# In[109]:


data = df[['Apps', 'Accept']]

data


# In[110]:


data = df[['Apps', 'Accept']].values

data


# # 35

# In[111]:


import scipy.cluster.hierarchy as sch


# In[112]:


fig = plt.figure(figsize=(12,8))

dendrogram = sch.dendrogram(sch.linkage(data, method='ward'))

plt.title('Dendrogram')
plt.xlabel('Students')
plt.ylabel('Euclidean distances')


# # 36

# In[113]:


from sklearn.cluster import AgglomerativeClustering


# In[114]:


hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(data)

y_hc


# # 37

# In[116]:


data[y_hc == 0]


# In[118]:


x1 = data[y_hc == 0][:,0]
y1 = data[y_hc == 0][:,1]
x2 = data[y_hc == 1][:,0]
y2 = data[y_hc == 1][:,1]


# In[119]:


fig = plt.figure(figsize=(12,8))
plt.scatter(x1, y1, s=20, c='blue', label='Cluster 1')
plt.scatter(x2, y2, s=20, c='red', label='Cluster 2')
plt.title('Cluster of Student with cluster=2')
plt.xlabel('Apps')
plt.ylabel('Accept')
plt.legend()
plt.grid()
plt.show()


# # 38

# In[122]:


data = df[['Top25perc','Outstate']]

data = np.array(data)

data


# # 39

# In[123]:


fig = plt.figure(figsize=(12,8))

dendrogram = sch.dendrogram(sch.linkage(data, method='ward'))

plt.title('Dendrogram')
plt.xlabel('Students')
plt.ylabel('Euclidean distances')


# # 40   ได้จำนวน cluster ที่เหมาะสมเท่ากับ 2

# In[124]:


hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(data)

y_hc


# # 41

# In[125]:


x1 = data[y_hc == 0][:,0]
y1 = data[y_hc == 0][:,1]
x2 = data[y_hc == 1][:,0]
y2 = data[y_hc == 1][:,1]


# In[127]:


plt.figure(figsize=(12,8))
plt.scatter(x1, y1, s=20, c='blue', label='Cluster 1')
plt.scatter(x2, y2, s=20, c='red', label='Cluster 2')
plt.title('Cluster of Student with cluster=2')
plt.xlabel('Top25perc')
plt.ylabel('Outstate')
plt.legend()
plt.grid()
plt.show()


# # 42  เปรียบเทียบ จำนวน cluster ของ KMeans กับ Hierarchy 

# In[128]:


#สรุปว่า KMeans เหมาะสมกว่าทั้งสองชุด

#โดยชุดแรกแบ่งเป็น 4 clusters ของความยากในการเข้ามหาลัย

#และชุดสองแบ่งเป็น 6 clusters ของจำนวนนักเรียน Top 25%


# # 43  ลองทำ Hierarchy โดยมี cluster = 6 ให้เหมือนกับ KMeans แล้วเอามาเทียบผลลัพธ์กัน

# In[129]:


hc = AgglomerativeClustering(n_clusters =6, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(data)

y_hc


# In[130]:


x1 = data[y_hc == 0][:,0]
y1 = data[y_hc == 0][:,1]
x2 = data[y_hc == 1][:,0]
y2 = data[y_hc == 1][:,1]
x3 = data[y_hc == 2][:,0]
y3 = data[y_hc == 2][:,1]
x4 = data[y_hc == 3][:,0]
y4 = data[y_hc == 3][:,1]
x5 = data[y_hc == 4][:,0]
y5 = data[y_hc == 4][:,1]
x6 = data[y_hc == 5][:,0]
y6 = data[y_hc == 5][:,1]


# In[131]:


fig = plt.figure(figsize=(12,8))
plt.scatter(x1, y1, s=20, c='blue', label='Cluster 1')
plt.scatter(x2, y2, s=20, c='red', label='Cluster 2')
plt.scatter(x3, y3, s=20, c='yellow', label='Cluster 3')
plt.scatter(x4, y4, s=20, c='pink', label='Cluster 4')
plt.scatter(x5, y5, s=20, c='orange', label='Cluster 5')
plt.scatter(x6, y6, s=20, c='purple', label='Cluster 6')
plt.title('Cluster of Student with cluster=6')
plt.xlabel('Top25perc')
plt.ylabel('Outstate')
plt.legend()
plt.grid()
plt.show()


# In[ ]:




