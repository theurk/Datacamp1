#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd


# # ข้อ 1

# In[13]:


df = pd.read_csv('train.csv')
df1 = pd.read_csv('test.csv')
df2 = pd.read_csv('gender_submission.csv')


# In[188]:


df


# In[189]:


df1


# In[190]:


df2


# In[191]:


df2['Survived']


# In[14]:


df1.insert(1, 'Survived', df2['Survived'])


# In[193]:


df1


# In[15]:


New_df = pd.concat([df, df1], ignore_index=True)
New_df1 = pd.concat([df, df1], ignore_index=True)


# In[326]:


New_df


# # ข้อ 2

# In[196]:


New_df.head(10)


# In[197]:


New_df.tail(10)


# In[198]:


New_df.sample(10)


# # ข้อ 3

# In[199]:


New_df['Embarked']


# In[200]:


New_df['Embarked'].dropna


# # ข้อ4

# In[201]:


avg = New_df['Age'].mean()

print(avg)


# In[203]:


New_df['Age'].fillna(value=avg, inplace=True)

New_df


# # ข้อ 5

# In[205]:


New_df['Age'] = 0

New_df


# # ข้อ 6

# In[125]:


New_df['Age'].dropna

New_df


# # ข้อ 7

# In[127]:


New_df['Sex'].value_counts()


# In[131]:


male = 843

female = 466

Ratio_male = (843/(male+female))*100

Ratio_male


# In[132]:


Ratio_female = (466/(male+female))*100

Ratio_female


# # ข้อ 8

# In[143]:


New_df[New_df['Sex'] == 'male']


# In[144]:


New_df[New_df['Survived'] == 1]


# In[151]:


Survived_male = New_df[(New_df['Sex'] == 'male') & (New_df['Survived'] == 1)]

Survived_male


# In[149]:


New_df[New_df['Sex'] == 'female']


# In[152]:


Survived_female = New_df[(New_df['Sex'] == 'female') & (New_df['Survived'] == 1)]

Survived_female


# In[154]:


Survived_male['Survived'].value_counts()


# ผู้ชายรอดชีวิต 109 คน

# In[155]:


Survived_female['Survived'].value_counts()


# ผู้หญิงรอดชีวิต 385 คน

# # ข้อ 9 

# In[167]:


All_Survived = New_df[New_df['Survived'] == 1]

All_Survived


# In[168]:


All_Survived.groupby('Sex').size()


# In[265]:


New_df['Sex'].value_counts()


# In[178]:


All_Survived.groupby('Sex').size() / New_df['Sex'].value_counts()*100


# # ข้อ 10

# In[181]:


New_df['Sex'].replace({'female':'0','male':'1'}, inplace=True)

New_df


# # ข้อ 11

# In[219]:


New_df1


# In[222]:


Passenger = New_df1[['Name', 'Age', 'Embarked']]

Passenger


# In[240]:


New_df1.groupby('Age').max()


# In[233]:


Passenger.iloc[630]


# # ข้อ 12

# In[234]:


Passenger.iloc[1245]


# # ข้อ 13

# In[245]:


Fared_Passenger = New_df1[['Fare', 'Name', 'Age', 'Survived']]

Fared_Passenger


# In[242]:


New_df.groupby('Fare').max()


# In[246]:


Fared_Passenger.iloc[1234]


# # ข้อ 14

# In[247]:


Fared_Passenger.iloc[1263]


# # ข้อ 15

# In[248]:


New_df['Pclass'].value_counts()


# # ข้อ 16

# In[250]:


New_df['Pclass'].mean()


# # ข้อ 17

# In[252]:


New_df1


# In[260]:


Age50_survived = New_df1[(New_df1['Age'] > 50) & (New_df1['Survived'] == 1)]

Age50_survived


# In[267]:


Age50_survived['Survived'].count()


# # ข้อ 18

# In[300]:


Pclass_Describe = New_df1['Pclass']

Pclass_Describe.describe()


# # ข้อ 19

# In[271]:


Pclass_price = New_df1[['Pclass', 'Fare']]

Pclass_price


# In[275]:


Pclass1_price = Pclass_price[Pclass_price['Pclass'] == 1]

Pclass1_price


# In[285]:


Pclass1_price.max()


# In[281]:


Pclass2_price = Pclass_price[Pclass_price['Pclass'] == 2]

Pclass2_price


# In[286]:


Pclass2_price.max()


# In[282]:


Pclass3_price = Pclass_price[Pclass_price['Pclass'] == 3]

Pclass3_price


# In[287]:


Pclass3_price.max()


# # ข้อ 20

# In[16]:


New_df = pd.concat([df, df1], ignore_index=True)

All_Age = New_df[['Age', 'Survived']]

All_Age


# In[21]:


All_Age_Completed = All_Age.dropna()

All_Age_Completed


# In[22]:


bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 120]
labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-80', '80+']
All_Age_Completed['Age Group'] = pd.cut(All_Age_Completed.Age, bins, labels = labels,include_lowest = True)

All_Age_Completed


# In[50]:


AllAge_0s = All_Age_Completed[All_Age_Completed['Age Group'] == '0-9']
AllAge_10s = All_Age_Completed[(All_Age_Completed['Age Group'] == '10-19')]
AllAge_20s = All_Age_Completed[(All_Age_Completed['Age Group'] == '20-29')]
AllAge_30s = All_Age_Completed[(All_Age_Completed['Age Group'] == '30-39')]
AllAge_40s = All_Age_Completed[(All_Age_Completed['Age Group'] == '40-49')]
AllAge_50s = All_Age_Completed[(All_Age_Completed['Age Group'] == '50-59')]
AllAge_60s = All_Age_Completed[(All_Age_Completed['Age Group'] == '60-69')]
AllAge_70s = All_Age_Completed[(All_Age_Completed['Age Group'] == '70-79')]
AllAge_80Plus = All_Age_Completed[(All_Age_Completed['Age Group'] == '80+')]

AllAge_0s['Survived'].count()


# In[51]:


Survived_Age_0s = All_Age_Completed[(All_Age_Completed['Age Group'] == '0-9') & (All_Age_Completed['Survived'] == 1)]
Survived_Age_10s = All_Age_Completed[(All_Age_Completed['Age Group'] == '10-19') & (All_Age_Completed['Survived'] == 1)]
Survived_Age_20s = All_Age_Completed[(All_Age_Completed['Age Group'] == '20-29') & (All_Age_Completed['Survived'] == 1)]
Survived_Age_30s = All_Age_Completed[(All_Age_Completed['Age Group'] == '30-39') & (All_Age_Completed['Survived'] == 1)]
Survived_Age_40s = All_Age_Completed[(All_Age_Completed['Age Group'] == '40-49') & (All_Age_Completed['Survived'] == 1)]
Survived_Age_50s = All_Age_Completed[(All_Age_Completed['Age Group'] == '50-59') & (All_Age_Completed['Survived'] == 1)]
Survived_Age_60s = All_Age_Completed[(All_Age_Completed['Age Group'] == '60-69') & (All_Age_Completed['Survived'] == 1)]
Survived_Age_70s = All_Age_Completed[(All_Age_Completed['Age Group'] == '70-79') & (All_Age_Completed['Survived'] == 1)]
Survived_Age_80Plus = All_Age_Completed[(All_Age_Completed['Age Group'] == '80+') & (All_Age_Completed['Survived'] == 1)]

Survived_Age_0s['Survived'].value_counts()


# In[57]:


Ratio_0s_Survived = (Survived_Age_0s['Survived'].value_counts()/AllAge_0s['Survived'].count())*100
Ratio_10s_Survived = (Survived_Age_10s['Survived'].value_counts()/AllAge_10s['Survived'].count())*100
Ratio_20s_Survived = (Survived_Age_20s['Survived'].value_counts()/AllAge_20s['Survived'].count())*100
Ratio_30s_Survived = (Survived_Age_30s['Survived'].value_counts()/AllAge_30s['Survived'].count())*100
Ratio_40s_Survived = (Survived_Age_40s['Survived'].value_counts()/AllAge_40s['Survived'].count())*100
Ratio_50s_Survived = (Survived_Age_50s['Survived'].value_counts()/AllAge_50s['Survived'].count())*100
Ratio_60s_Survived = (Survived_Age_60s['Survived'].value_counts()/AllAge_60s['Survived'].count())*100
Ratio_70s_Survived = (Survived_Age_70s['Survived'].value_counts()/AllAge_70s['Survived'].count())*100
Ratio_80Plus_Survived = (Survived_Age_80Plus['Survived'].value_counts()/AllAge_80Plus['Survived'].count())*100

print(Ratio_0s_Survived)
print(Ratio_10s_Survived)
print(Ratio_20s_Survived)
print(Ratio_30s_Survived)
print(Ratio_40s_Survived)
print(Ratio_50s_Survived)
print(Ratio_60s_Survived)
print(Ratio_70s_Survived)
print(Ratio_80Plus_Survived)


# # ข้อ21

# In[303]:


New_df1['Fare'].sort_values()


# In[304]:


New_df1['Fare'].sort_values()[::-1]


# # ข้อ 22

# In[373]:


New_df = pd.concat([df, df1], ignore_index=True)


# In[374]:


New_df['Name']

New_df


# In[375]:


New_LastName = New_df['Name'].str.split(", ", n = -1, expand = True)

New_df["Last Name"] = New_LastName[0]

New_Name = New_df['Name'].str.split(" ", n = -1, expand = True)

New_df["Title"] = New_Name[1]

New_df["First Name"] = New_Name[2]

New_df["Middle Name"] = New_Name[3]

New_df


# In[382]:


Only_LastName = New_df['Last Name']

Only_LastName


# In[385]:


Same_LastName = Only_LastName[Only_LastName.duplicated()]


# In[386]:


Same_LastName.value_counts()


# # ข้อ 23

# In[388]:


Only_LastName


# In[389]:


Only_LastName.unique()


# In[390]:


Only_LastName.nunique()


# In[ ]:




