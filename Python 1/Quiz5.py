#!/usr/bin/env python
# coding: utf-8


lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

result = map(lambda x:x**3, lst)
print(list(result))


# In[9]:


# ต่อไปเป็นแบบไม่ได้ตั้งตัวแปร


result = map(lambda x:x**3, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(list(result))


# In[11]:


# ต่อไปเป็นแบบ function ธรรมดา


def Meaow(a):
    return a**3
print(list(map(Meaow, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))


# In[14]:


lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(list(map(Meaow, lst)))


# In[15]:


# filter แบบ lambda ยังไม่ตั้งตัวแปร


final_list = filter(lambda x:((x**2)%2 != 0), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(list(final_list))


# In[19]:


# filter แบบ lambda และตั้งตัวแปร


lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
final_list = filter(lambda x:((x**2)%2 != 0), lst)
print(list(final_list))


# In[21]:


# filter แบบ function ธรรมดา


def final_list(a):
    return (a**2)%2 != 0
print(list(filter(final_list, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))


# In[23]:


lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(list(filter(final_list, lst)))

# In[24]: