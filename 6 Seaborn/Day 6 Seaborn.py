#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# #1

# In[2]:


Names = ['A','B','C','D']

Years = ['2014 (หน่วยล้านบาท)','2015 (หน่วยล้านบาท)','2016 (หน่วยล้านบาท)','2017 (หน่วยล้านบาท)']

Values = np.random.uniform(50,500,(4,4))

print(Values)


# In[3]:


df = pd.DataFrame(Values,Names,Years)
df


# In[5]:


#2


# In[6]:


df.index


# In[7]:


df.values


# In[8]:


df.mean(axis=1)


# In[9]:


sns.barplot(x=df.index,y=df.mean(axis=1),data=df)


# In[10]:


sns.barplot(x=df.mean(axis=1),y=df.index, data=df)


# In[11]:


#3


# In[12]:


sns.scatterplot(x=df.index,y=df['2017 (หน่วยล้านบาท)'], data=df)


# In[13]:


sns.scatterplot(x=df['2017 (หน่วยล้านบาท)'],y=df.index, data=df)


# In[14]:


#4


# In[15]:


df = pd.read_csv('train.csv')

df


# In[16]:


sns.countplot(x='Embarked', data=df)


# In[17]:


sns.countplot(y='Embarked', data=df)


# In[18]:


#5


# In[19]:


sns.countplot(x='Pclass', data=df)


# In[20]:


sns.countplot(y='Pclass', data=df)


# #6

# In[21]:


fig = plt.figure(figsize=(10,8))
sns.boxplot(x='Sex',y='Fare',data=df)


# In[22]:


#7


# In[23]:


sns.boxplot(x='Pclass',y='Fare',data=df)


# In[24]:


#8


# In[25]:


sns.barplot(x='Embarked',y='Fare',data=df)


# In[26]:


#9


# In[27]:


sns.barplot(x='Pclass',y='Age',data=df)


# In[28]:


#10


# In[29]:


sns.stripplot(x='Pclass',y='Fare',data=df)


# In[30]:


#11


# In[32]:


sns.stripplot(x='Survived',y='Age',data=df, hue='Sex')


# In[34]:


sns.stripplot(x='Survived',y='Age',data=df, hue='Sex',dodge=True)


# In[33]:


#12


# In[38]:


sns.stripplot(x='Sex',y='Age',data=df)


# In[39]:


#13


# In[40]:


New_df = pd.crosstab(df['Pclass'],df['Sex'])

New_df


# In[44]:


fig = plt.figure(figsize=(10,8))
sns.heatmap(New_df, cmap='coolwarm', annot=True)


# In[45]:


#14


# In[46]:


New_df2 = df.pivot_table(index='Pclass',columns='Sex',values='Fare')

New_df2


# In[48]:


sns.heatmap(New_df2, cmap='coolwarm')


# In[49]:


#15


# In[50]:


sns.heatmap(New_df2, cmap='coolwarm',linecolor='white',linewidth=2)


# In[51]:


#16


# In[53]:


df_corr = df.corr()

df_corr


# In[54]:


sns.heatmap(df_corr, cmap='coolwarm',linecolor='white',linewidth=2)


# In[55]:


#17


# In[61]:


sns.clustermap(df_corr,cmap='coolwarm',linecolor='white',linewidth=1.5, annot=True)


# In[ ]:


#18


# In[60]:


sns.scatterplot(x=df['SibSp'],y=df['Parch'],data=df)


# In[62]:


#19


# In[68]:


sns.scatterplot(x=df['Pclass'],y=df['Age'],data=df,hue=df['Sex'])


# In[69]:


#20


# In[77]:


sns.distplot(df['Fare'])


# In[82]:


df['Age'] = pd.to_numeric(df['Age'])

df


# In[89]:


newwww = df.dropna()

newwww


# In[90]:


sns.distplot(newwww['Age'])


# In[75]:


#21


# In[76]:


sns.pairplot(df)


# In[ ]:




