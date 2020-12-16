#!/usr/bin/env python
# coding: utf-8

# # Project 1 Part 1

# # 1

# In[1]:


Count = 0

Sum = 0

number = float(input('ใส่ตัวเลขใดๆ (ใส่ 0.0 เพื่อจบการทำงาน):'))

while number != 0:
    Sum = Sum + number
    Count = Count + 1
    number = float(input('ใส่ตัวเลขใดๆ (ใส่ 0.0 เพื่อจบการทำงาน):'))
    
Average = Sum/Count

print(Average*Count)
print('Finish')


# # 2

# In[5]:


low_bar = int(input('ใส่ขอบเขตล่าง: '))
high_bar = int(input('ใส่ขอบเขตบน: '))

count = 0

for number in range(low_bar, high_bar + 1):
    if number > 1:
        for i in range(2, number):
            if number %i == 0:
                break
        else:
            count = count + 1
            print('%d is a prime number' %(number))
            
print('จำนวน Prime number ทั้งหมด: ', count)


# # Project 1 Part 2 เรื่องยอดขายเกม

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # 1

# In[8]:


df = pd.read_csv('vgsales.csv')
df


# In[9]:


df.info()


# In[10]:


df.describe()


# # 2

# In[11]:


df.head(10)


# In[12]:


df.tail(10)


# In[13]:


df.sample(10)


# # 3

# In[14]:


df['Platform'].value_counts()


# In[15]:


df['Platform'].value_counts().head(10)


# # 4

# In[16]:


df['Platform'].value_counts().tail(10)


# # 5

# In[17]:


df['Genre'].value_counts().head(10)


# # 6

# In[18]:


df['Genre'].value_counts().tail(10)


# # 7

# In[19]:


df[df['Name']=='Grand Theft Auto V']


# # 8

# In[21]:


df[df['Name'].duplicated()]


# In[22]:


df[df['Name'].duplicated(keep=False)]


# In[23]:


df[df['Name'].duplicated(keep=False)].sort_values('Name')


# # 9

# In[27]:


df[df[['Name', 'Platform']].duplicated(keep=False)]


# # 10

# In[28]:


df.drop(df.index[[4145,14999,16127]], inplace = True)


# # 11

# In[31]:


df.groupby('Name').sum().sort_values('Global_Sales')[::-1]


# In[32]:


df2 = df.groupby('Name').sum().reset_index()

df2


# In[33]:


df2[df2['Name']=='FIFA 15']


# # 12

# In[35]:


df[df['Name']=='Grand Theft Auto V'].groupby('Name').sum()


# # 13

# In[37]:


df2 = df['Name'].value_counts()

df2 = pd.DataFrame(df2)

df2


# In[41]:


df2 = df2[df2['Name'] > 1]

df2


# In[43]:


df2.rename(columns={'Name':'Count'})


# # 14

# In[44]:


df.groupby('Publisher').sum()


# In[45]:


df.groupby('Publisher').sum().sort_values('Global_Sales')[::-1]


# # 15

# In[48]:


df_CallofDuty = df[df['Name'].apply(lambda check: check[0:12]=='Call of Duty')]

df_CallofDuty


# # 16

# In[51]:


df_CallofDuty[df_CallofDuty['Platform']=='PC'].sort_values('EU_Sales')[::-1].head(5)


# # 17

# In[54]:


df.groupby('Platform').sum().sort_values('EU_Sales')[::-1]


# # 18

# In[55]:


df.groupby('Genre').sum().sort_values('Global_Sales')[::-1]


# # 19

# In[57]:


fig = plt.figure(figsize=(18,12))
sns.barplot(x='Platform', y='Global_Sales', data=df)


# # 20

# In[60]:


df2 = df.groupby('Publisher').sum().sort_values('Global_Sales')[::-1]

df2 = df2.head(5)

df2


# In[61]:


fig = px.pie(df2, values='Global_Sales', names=df2.index)

fig.show()


# # 21

# In[63]:


fig = plt.figure(figsize=(18,12))
sns.countplot(x='Genre', data=df)


# # 22

# In[65]:


df3 = df_CallofDuty[df_CallofDuty['Platform']=='X360']


# In[67]:


fig = plt.figure(figsize=(18,12))
sns.barplot(x='Name', y='Global_Sales', data=df3.head(5))


# # 23

# In[68]:


df4 = df.groupby('Year').sum()

df4


# In[69]:


fig = px.line(df4, x=df4.index, y='NA_Sales', title='Sales of North America')

fig.show()


# # 24

# In[73]:


fig = plt.figure(figsize=(18,12))
sns.stripplot(x='Genre', y='Global_Sales', data=df)


# # 25

# In[75]:


fig = plt.figure(figsize=(18,12))
sns.distplot(df['Year'].dropna())


# # 26

# In[76]:


fig = plt.figure(figsize=(18,12))
sns.barplot(x=df4.index, y='JP_Sales', data=df4)
fig.autofmt_xdate()


# # Project 1 Part 2 เรื่อง Air BNB

# # 1

# In[77]:


df = pd.read_csv('AB_NYC_2019.csv')

df


# # 2

# In[78]:


df.head(10)


# In[79]:


df.tail(10)


# In[80]:


df.sample(10)


# # 3

# In[81]:


df['neighbourhood'].value_counts().head(10)


# # 4

# In[82]:


df['neighbourhood'].value_counts().tail(10)


# # 5

# In[83]:


df['neighbourhood_group'].value_counts().head(10)


# # 6

# In[84]:


df['neighbourhood_group'].value_counts().tail(10)


# # 7

# In[86]:


df.groupby('neighbourhood_group').mean()['price']


# # 8

# In[87]:


df.groupby('room_type').mean()['price']


# # 9

# In[88]:


df2 = df['neighbourhood'].value_counts()

df2 = pd.DataFrame(df2)

df2


# In[89]:


df2[df2['neighbourhood']==1]


# # 10

# In[90]:


df['room_type'].value_counts()


# # 11

# In[92]:


df.groupby('neighbourhood').sum().sort_values('number_of_reviews')[::-1].head(10)


# In[93]:


df.groupby('neighbourhood_group').sum().sort_values('number_of_reviews')[::-1].head(10)


# ## 12

# In[94]:


df.sort_values('minimum_nights')[::-1]


# In[96]:


df.sort_values('minimum_nights')[::-1].drop_duplicates('neighbourhood_group').head(3)


# # 13

# In[97]:


df.groupby(['host_id','host_name']).mean()


# In[98]:


df.groupby(['host_id','host_name']).mean().reset_index()


# In[102]:


df2 = df.groupby(['host_id','host_name']).mean().reset_index().sort_values('calculated_host_listings_count')[::-1]

df2[['host_id','host_name','calculated_host_listings_count']].head(10)


# # 14

# In[104]:


df2 = df.groupby(['host_id','host_name']).size().reset_index(name='Count')

df2.sort_values('Count')


# In[105]:


df2['host_name'].value_counts().head(10)


# # 15

# In[107]:


df2 = df.groupby(['host_id','host_name'], as_index=False).sum()
df2


# In[109]:


df2.sort_values('number_of_reviews')[::-1].head(10)


# # 16

# In[110]:


df3 = df.groupby(['host_id','host_name'], as_index=False).mean()
df3


# In[112]:


df3.sort_values('price')[::-1].head(10)


# # 17

# In[114]:


df2 = df

df2


# In[115]:


df2['last_review'][0].split('-')[0]


# In[116]:


df2 = df2.dropna()


# In[117]:


df2['Year-LR'] = df2['last_review'].apply(lambda x: x.split('-')[0])
df2['Month-LR'] = df2['last_review'].apply(lambda x: x.split('-')[1])
df2['Day-LR'] = df2['last_review'].apply(lambda x: x.split('-')[2])
df2


# # 18

# In[119]:


day_of_week = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


# In[120]:


df2['last_review'] = pd.to_datetime(df2['last_review'])
df2


# In[121]:


df2['last_review'][0].dayofweek


# In[123]:


df2['day_of_week'] = df2['last_review'].apply(lambda time: time.dayofweek)

df2


# In[124]:


df2['day_of_week'] = df2['day_of_week'].map(day_of_week)

df2


# # 19

# In[125]:


sns.countplot(x='room_type', data=df)


# # 20

# In[128]:


df2 = df.groupby('neighbourhood').size().reset_index(name = 'Count').sort_values('Count')[::-1].head(5)

df2


# In[129]:


fig = px.pie(df2, values='Count', names='neighbourhood')
fig.show()


# # 21

# In[130]:


sns.boxplot(x='neighbourhood_group', y='number_of_reviews', data=df)


# # 22

# In[131]:


df.corr()


# In[132]:


sns.heatmap(df.corr())


# # 23

# In[157]:


df2 = df

df2


# In[158]:


df2 = df2.dropna()


# In[159]:


df2['Year-LR'] = df2['last_review'].apply(lambda x: x.split('-')[0])
df2['Month-LR'] = df2['last_review'].apply(lambda x: x.split('-')[1])
df2['Day-LR'] = df2['last_review'].apply(lambda x: x.split('-')[2])
df2


# In[160]:


df2['last_review'] = pd.to_datetime(df2['last_review'])
df2


# In[161]:


df2['day_of_week'] = df2['last_review'].apply(lambda time: time.dayofweek)

df2


# In[162]:


df2['day_of_week'] = df2['day_of_week'].map(day_of_week)

df2


# In[163]:


#เริ่มทำ

df3 = df2[df2['Year-LR']=='2018']

df3


# In[165]:


df3 = df3.groupby('Month-LR').sum()

df3


# In[166]:


sns.barplot(x=df3.index, y='number_of_reviews', data=df3)


# # 24

# In[155]:


fig = px.pie(df2, values='number_of_reviews', names='day_of_week')
fig.show()


# # 25

# In[167]:


df3 = df2.groupby('Month-LR').mean().reset_index()
df3


# In[168]:


fig = px.line(df3, x='Month-LR', y='price')
fig.show()


# # 26

# In[170]:


import datetime

today = datetime.datetime.today()
today


# In[172]:


oldday = df2['last_review'][0]

oldday


# In[173]:


(today - oldday).days


# In[174]:


df2['Diff'] = df2['last_review'].apply(lambda past: (today-past).days)


# In[175]:


df2


# # 27

# In[176]:


df5 = df2.groupby('neighbourhood_group').mean().reset_index()
df5


# In[177]:


fig = px.pie(df5, values='Diff', names='neighbourhood_group')
fig.show()


# # 28

# In[178]:


df6 = df2.groupby('neighbourhood').mean().reset_index()
df6


# In[179]:


df6 = df6.sort_values('Diff')[::-1].head(10)
df6


# In[181]:


plt.figure(figsize=(16,8))
sns.barplot(x='neighbourhood', y='Diff', data=df6)


# # 29

# In[182]:


df7 = df2.groupby('neighbourhood').mean().reset_index().sort_values('Diff')
df7


# In[183]:


sns.barplot(x='neighbourhood', y='Diff', data=df7)


# In[185]:


plt.figure(figsize=(16,8))
sns.barplot(x='neighbourhood', y='Diff', data=df7.head(10))


# # 30

# In[186]:


fig = px.scatter(df2, x='Diff', y='minimum_nights')
fig.show()


# In[ ]:




