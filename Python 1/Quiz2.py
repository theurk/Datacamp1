#!/usr/bin/env python
# coding: utf-8


# ข้อ 1
# In[ ]:


duck = lambda x: x/3
print(duck(6))


# In[ ]:


# ข้อ 2


def dog(w, x, y, z):
    result = (w+x+y+z)/4
    return result
mean = dog(1, 2, 3, 4)
print(mean)


# In[15]:


# ข้อ 3


mean = lambda x, y, z: (x+y+z)/3
print(mean(6, 7, 8))

# In[20]: