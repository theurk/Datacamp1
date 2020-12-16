#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # ข้อ 1

# In[2]:


x = [1,2,3,4,5,6,7]
x = np.array(x)
y = x**2


# In[8]:


plt.title("Graph y = x**2")
plt.xlabel('1-7')
plt.ylabel('x**2')

plt.plot(x,y)

plt.show()


# # ข้อ 2

# In[10]:


x = np.arange(0,50)
y = np.sin(x)


# In[13]:


plt.title("y = sin(x)")
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x,y,c='r')

plt.show()


# # ข้อ 3

# In[15]:


y = [1,2,3,4,5,6,7]
y = np.array(y)
x = y**2


# In[16]:


plt.title("Graph x = y**2")
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x,y,c='b')

plt.show()


# # ข้อ 4

# In[30]:


x = np.arange(15,50)
y = np.cos(x)

plt.title("y = cos(x)")
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x,y,c='r',ls='--')

plt.show()


# # ข้อ 5

# In[53]:


x = np.arange(0,7)
y1 = x**2
y2 = x**3

plt.title("y1 = x**2 and y2 = x**3")
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x,y1,c='r',marker='o')
plt.plot(x,y2,c='b',marker='^')

plt.show()


# # ข้อ 6

# In[56]:


x = np.arange(0,7)
y = np.arange(0,7)
x = y**2
y = x**2


plt.title("x = y**2 and y = x**2")
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x,x,c='r',marker='o',label='x = y**2')
plt.plot(x,y,c='b',marker='^',label='y = x**2')

plt.legend(loc=0)

plt.show()


# In[55]:


x = np.arange(0,7)
y = np.arange(0,7)


plt.title("x = y**2 and y = x**2")
plt.xlabel('x')
plt.ylabel('y')

plt.plot(y**2,y,c='r',marker='o',label='x = y**2')
plt.plot(x,x**2,c='b',marker='^',label='y = x**2')

plt.legend(loc=0)

plt.show()


# # ข้อ 7

# In[64]:


x = np.arange(0,20)
x = y**2 + 4*y

plt.title("x = y**2 + 4*y")
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x,y,c='#FFFF99')

plt.xlim([0,500000])
plt.ylim([0,1000])

plt.show()


# # ข้อ 8

# In[82]:


x = np.arange(0,20)
y = np.cos(x)
fig = plt.figure()

plt.plot(x,x**2,c='r',label='x=x**2')

plt.title("2 graph")
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=0)

ax1 = fig.add_axes([0.2,0.45,0.3,0.3])
ax1.plot(x,y,label='y=cos(x)')

plt.legend(loc=2)

plt.show()


# # ข้อ 9

# In[84]:


import pandas as pd


# In[85]:


df = pd.read_csv('train.csv')


# In[86]:


df


# In[89]:


Ticket_class = df['Pclass']

Ticket_class


# In[99]:


Ticket_class.value_counts()


# In[101]:


Ticket_ratio = Ticket_class.value_counts()/Ticket_class.count()*100

Ticket_ratio


# In[106]:


plt.figure(figsize=[5,5])
plt.pie(Ticket_ratio,labels=Ticket_class.unique())


# # ข้อ 10

# In[113]:


Port = df['Embarked']

Port


# In[114]:


Port_ratio = Port.value_counts()/Port.count()*100

Port_ratio


# In[118]:


plt.figure(figsize=[5,5])
explode = [0,0,0.1]
plt.pie(Port_ratio,labels=['Southampton','Cherbourg','Queenstown'],explode=explode)


# # ข้อ 11

# In[158]:


Ticket_price = df['Fare']

Ticket_price


# In[162]:


plt.hist(Ticket_price)

plt.show()


# # ข้อ 12

# In[165]:


df1 = df[['Age','Fare']]

df1


# In[169]:


x = df['Age']
y = df['Fare']

plt.figure(figsize=[5,5])
plt.scatter(x,y,c='r')
plt.title('Scatter plot')
plt.xlabel('Age')
plt.ylabel('Fare')

plt.show()


# # ข้อ 13

# In[171]:


df2 = df[['Pclass','Fare']]

df2


# In[182]:


Ticket_Mean = df2.groupby('Pclass').mean()

Ticket_Mean


# In[196]:


x = Ticket_Mean.index
y = Ticket_Mean['Fare']


plt.figure(figsize=[5,5])
plt.bar(x,y)
plt.title('Bar chart')
plt.xlabel('Pclass')
plt.ylabel('Fare')

plt.show()


# # ข้อ 14

# In[198]:


df3 = df[['Age','Fare']]

df3


# In[199]:


AgeTicket_Mean = df3.groupby('Age').mean()

AgeTicket_Mean


# In[200]:


x = AgeTicket_Mean.index
y = AgeTicket_Mean['Fare']


plt.figure(figsize=[5,5])
plt.bar(x,y)
plt.title('Bar chart')
plt.xlabel('Age')
plt.ylabel('Fare')

plt.show()


# # ข้อ 15

# In[203]:


df


# In[204]:


df4 = df['Name']

df4


# In[205]:


LastName = df['Name'].str.split(", ", n = -1, expand = True)

df["Last Name"] = LastName[0]

Name = df['Name'].str.split(" ", n = -1, expand = True)

df["Title"] = Name[1]

df["First Name"] = Name[2]

df["Middle Name"] = Name[3]

df


# In[206]:


Only_LastName = df['Last Name']

Only_LastName


# In[214]:


Same_LastName = Only_LastName[Only_LastName.duplicated()]

Same_LastName


# In[218]:


counts = Same_LastName.value_counts()

counts


# In[219]:


x = counts.index
y = counts


plt.figure(figsize=[5,5])
plt.bar(x,y)
plt.title('Bar chart')
plt.xlabel('Last Name')
plt.ylabel('Frequency')

plt.show()


# # ข้อ 16

# In[225]:


df5 = df[['Sex','Age','Fare']]

df5


# In[230]:


AgeTicket_Mean = df5.groupby(['Age']).mean()

AgeTicket_Mean


# In[259]:


x = AgeTicket_Mean.index
y = AgeTicket_Mean['Fare']
y1 = x+y
y2 = x*y


fig = plt.figure()
ax1 = fig.add_subplot(321,title='Num 1')

ax1.plot(x,y,'r')

ax2 = fig.add_subplot(222,title='Num 2')
ax2.plot(y,x,'m')

ax3 = fig.add_subplot(323,title='Num 3')
ax3.plot(y1,np.sin(x+2),'b')

ax4 = fig.add_subplot(224,title='Num 4')
ax4.plot(x**2,y2,'y')

ax5 = fig.add_subplot(325,title='Num 5')
ax5.plot(x/2,y1/y)


# In[ ]:




