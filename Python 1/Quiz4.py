#!/usr/bin/env python
# coding: utf-8


a = [[[1, 3], [3, 4]], [5, [5, 6], [7, 8]]]


a[1][1][1]


# In[3]:


# 2



Urk = ['อัครพล', 'เอกอรรถพร', 30]
Yves = ['ณัฐณิชา', 'ฤกษ์วิธี', 29 ]
Ae = ['อาบพร', 'เอกอรรถพร',  29]
Aum = ['กฤษฎิ์', 'สัมภวะผล', 24]


name_all = [Urk, Yves, Ae, Aum]


# In[8]:


print(name_all)


# In[9]:


del name_all[3]
print(name_all)


# In[10]:


Aoi = ['อภิสา', 'วงศ์สาคร', 58]


# In[11]:


name_all.insert(0, Aoi)
print(name_all)


# In[12]:


name_all[1] = ['อัครพล', 'เอกอรรถพร', 30, 'น้ำหนัก +10']
print(name_all)

# In[13]: