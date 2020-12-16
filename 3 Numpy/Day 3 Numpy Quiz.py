#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# 1

np.zeros(10)


# In[4]:


np.zeros((4,4))


# In[5]:


# 2

np.ones(5)


# In[6]:


np.ones((3,3))


# In[42]:


# 3

x = np.random.uniform(10,99,20)

x


# In[43]:


y = x**2

y


# In[49]:


# 4

p = np.random.uniform(10,99,20)

randomFloat = []

[p**2 for p in np.random.uniform(10,99,20)]

randomFloat.append(p)

print(randomFloat)


# In[62]:


# 5


# numpy

import time

time1 = time.time()

x = np.random.uniform(10,100,20) **2

print(x)

time2 = time.time()

print(time2 - time1)


# In[78]:


# list comprehension

import time

time1 = time.time()

z = range(10,100)

randomFloat = []

import random

[float(z)**2 for z in random.sample(z,20)]

randomFloat.append(p)

print(randomFloat)

time2 = time.time()

print(time2 - time1)


# In[66]:


# 6

np.arange(10,101)


# In[68]:


# 7

np.arange(10,2001,2)


# In[81]:


# 8

a = np.arange(12,901)

b = a%2 != 0

a[b]


# In[82]:


# 9

np.eye(4)


# In[85]:


# 10

np.linspace(5,25,100)


# In[87]:


# 11

np.random.rand(1)


# In[91]:


# 12

A = np.random.randint(50,100,(4,4))

A


# In[92]:


A.T


# In[105]:


# 13 - 17

arr = np.arange(1,26)

arr2 = arr.reshape(5,5)

arr2


# In[106]:


# 13

arr2[1:3,2:4]


# In[107]:


# 14

arr2[3:4,2:]


# In[108]:


# 15

arr2[3:,:2]


# In[119]:


# 16

a = arr2[1:3,2:4]
b = arr2[3]
c = arr2[4,:2]

print(a)

print(b)

print(c)


# In[120]:


# 17

arr = np.arange(1,26)

arr2 = arr.reshape(5,5)

arr2


# In[149]:


# 18

a = np.random.randint(10,101,(5,5))

b = []

print(a)

for a[i] in np.random.randint(10,101,(5,5)):
    
    for a[j] in np.random.randint(10,101,(5,5)):
        
        if a[i][j] > 50:
            
            b.append(a[i][j])
print('')    
print(b)


# In[150]:


# 19

a = np.random.randint(10,101,(5,5))

print(a)

b = a > 40
            

print('')  

a[b]


# In[265]:


# 20


import numpy as np

lst = []

i = 0

while i < 9:
    
    a = int(input('กรุณาระบุตัวเลขที่ต้องการ : '))
    
    if 10 <= a <= 100:
        
        if a in lst == a:             #ทำตรงนี้ไม่ได้ครับ ทำให้เลขไม่ซ้ำไม่ได้ครับ
            print('ตัวเลขห้ามซ้ำกัน')        #หมดปัญญาครับ ทำข้อเดียวเป็นครึ่งวัน ช่วยชี้แนะทีครับ
        else:
            lst.append(a)
            i = i+1
                
    else:
        print('ตัวเลขต้องอยู่ในช่วง 10-100 เท่านั้น')
        
        
print(lst)
        
lst1 = np.array([lst])

lst2 = lst1.reshape(3,3)

print(lst2)

lst3 = lst2.T

print(lst3)

lst3[2,2]

# In[265]:
# 

# 
