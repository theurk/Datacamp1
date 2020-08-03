#!/usr/bin/env python
# coding: utf-8


lst = [5, 2, 1, 3, 2]


from functools import reduce
multiply = reduce(lambda x, y: x**y, lst)
print(multiply)


# In[4]:


lst = [5, 2, 1, 3, 2]


from functools import reduce
def test(x, y):
    return x**y
multiply = reduce(test, lst)
print(multiply)


# In[12]:


lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
from functools import reduce
def test2(x, y):
    return x*y
multiple = reduce(test2, lst)
print(multiple)


# In[13]:


lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
from functools import reduce
multiple = reduce(lambda x,y: x*y, lst)
print(multiple)


# In[15]:




