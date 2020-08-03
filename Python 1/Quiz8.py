#!/usr/bin/env python
# coding: utf-8


A = [1, 2, 3, 4, 5]
B = [2, 3, 1, 3, 2]


#แบบที่ 1


C = map(lambda a, b: b**a, A, B)
print(list(C))


# In[7]:


#แบบที่ 2


A = [1, 2, 3, 4, 5]
B = [2, 3, 1, 3, 2]


# In[11]:


def ohmygod(a, b):
    return b**a

C = map(ohmygod, A, B)
print(list(C))


# In[ ]:


#เรื่อง zip


student_name = ['เอิ๊ก', 'เอ๊', 'อ้ำ']
student_number = ['32', '31', '38']
final_score = [100, 70, 50]


# In[13]:


student_data = list(zip(student_name, student_number, final_score))
print(student_data)

# In[14]: