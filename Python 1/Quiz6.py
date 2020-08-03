#!/usr/bin/env python
# coding: utf-8

# In[ ]:

lst = [1, 2, 3, 4, 5]

lst2 = [(x*2)-1 for x in lst]
print(lst2)


# In[3]:


lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
eiei = [item**3 for item in lst if item%3 != 0]
print(eiei)

# In[4]: