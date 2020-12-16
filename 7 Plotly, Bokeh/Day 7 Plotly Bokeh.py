#!/usr/bin/env python
# coding: utf-8

# In[75]:


import pandas as pd
import numpy as np
import plotly.express as px


# In[76]:


#1


# In[77]:


# FV = Future Value
# PV = Present Value
# i = อัตราการขึ้นเงินเดือน
# n = จำนวนปี/

# FV = PV*(1+i)**n
# PV = FV/(1+i)**n


# In[78]:


Names = ['A','B','C','D','E']

Years = ['2016','2017','2018','2019','2020']

Now = ['2020']

FV = np.array([10000,20000,30000,40000,60000])

PV = FV/(1+0.07)**n

    
    
for i in FV:
    
    PV2016 = FV/(1+0.07)**5
    PV2017 = FV/(1+0.07)**4
    PV2018 = FV/(1+0.07)**3
    PV2019 = FV/(1+0.07)**2
    


print(PV2016)
print(PV2017)
print(PV2018)
print(PV2019)


# In[79]:


All_Value = np.concatenate((PV2016, PV2017, PV2018, PV2019, FV))

All_Value


# In[80]:


New_All_Value = np.reshape(All_Value, (5, 5))

New_All_Value


# In[81]:


New_All_Value_T = New_All_Value.T

New_All_Value_T


# In[82]:


df = pd.DataFrame(New_All_Value_T,Names,Years)
df


# In[83]:


#2


# In[94]:


fig = px.bar(df, x=df.index, y=df['2017']*2)

fig.update_layout(xaxis_tickangle=-45)

fig.show()


# In[95]:


fig = px.bar(df, x=df['2017']*2, y=df.index)

fig.update_layout(xaxis_tickangle=-45)

fig.show()


# In[96]:


#3


# In[116]:


df_mean = df.mean(axis=1) 

df_mean


# In[129]:


fig = px.bar(df, x=df.index, y=df['2017']*2, 
             hover_data=['2016','2018','2019','2020', df_mean], 
             color='2016', labels={'index':'รายชื่อ', 'y':'เงินเดือนปี 2017'})

fig.update_layout(xaxis_tickangle=-45)

fig.show()


# In[130]:


#4


# In[133]:


df = pd.read_csv('train.csv')

df


# In[138]:


Age_GB_Mean = df.groupby('Age').mean()

Age_GB_Mean


# In[143]:


fig = px.line(df, x=Age_GB_Mean.index, y=Age_GB_Mean['Fare'], title='ราคาตั๋วโดยสารแบ่งตามอายุ',
             labels={'x':'อายุ', 'y':'ราคาตั๋ว'})

fig.show()


# In[144]:


#5


# In[145]:


Pclass_GB_Mean = df.groupby('Pclass').mean()

Pclass_GB_Mean


# In[151]:


fig = px.line(df, x=Pclass_GB_Mean.index, y=Pclass_GB_Mean['Fare'], title='ราคาตั๋วโดยสารแบ่งตามระดับชั้นโดยสาร',
             labels={'x':'ระดับชั้นโดยสาร', 'y':'ราคาตั๋ว'})

fig.show()


# In[147]:


#6


# In[154]:


fig = px.pie(df, values=df['Fare'], names=df['Age'], title='ราคารวมค่าโดยสารแบ่งตามอายุ')

fig.show()


# In[155]:


#7


# In[157]:


fig = px.pie(df, values=Age_GB_Mean['Fare'], names=Age_GB_Mean.index, title='ราคาเฉลี่ยค่าโดยสารแบ่งตามอายุ')

fig.show()


# In[158]:


#8


# In[160]:


df['Name']


# In[161]:


LastName = df['Name'].str.split(", ", n = -1, expand = True)

df["Last Name"] = LastName[0]

Name = df['Name'].str.split(" ", n = -1, expand = True)

df["Title"] = Name[1]

df["First Name"] = Name[2]

df["Middle Name"] = Name[3]

df


# In[163]:


Mean_Same_Lastname = df.groupby('Last Name').mean()

Mean_Same_Lastname


# In[164]:


fig = px.pie(df, values=Mean_Same_Lastname['Fare'], 
             names=Mean_Same_Lastname.index,
             title='ราคาเฉลี่ยค่าโดยสารแบ่งตามนามสกุล')

fig.show()


# In[165]:


#9


# In[171]:


fig = px.pie(df, values=Mean_Same_Lastname['Age'], 
             names=Mean_Same_Lastname.index,
             title='ค่าเฉลี่ยอายุของคนที่นามสกุลซ้ำกัน',
            color_discrete_sequence=px.colors.sequential.RdBu)

fig.show()


# In[168]:


#10


# In[175]:


fig = px.pie(df, values=df.index, 
             names=df['Embarked'],
             title='ค่าเฉลี่ยของแต่ละจุดหมายปลายทาง',
            color_discrete_sequence=px.colors.sequential.Purp)

fig.show()


# In[174]:


#11


# In[181]:


fig = px.scatter(Age_GB_Mean, x=Age_GB_Mean.index, y='Fare', 
                size='SibSp', title='บับเบิ้ลชาร์ท อิอิ',
             labels={'x':'อายุ', 'y':'ราคาตั๋ว'}, color='Pclass')

fig.show()


# In[180]:


#12


# In[182]:


fig = px.box(df, x='Sex', y='Fare')

fig.show()


# In[183]:


#13


# In[184]:


fig = px.box(df, x='Pclass', y='Fare')

fig.show()


# In[185]:


#14


# In[187]:


fig = px.box(df, x='Pclass', y='Age', points='all')

fig.show()


# In[188]:


#15


# In[189]:


fig = px.box(df, x='Survived', y='Age', points='all', color='Sex')

fig.show()


# In[190]:


#16


# In[191]:


EmBarked_Std = df.groupby('Embarked').std()

EmBarked_Std


# In[198]:


fig = px.bar(EmBarked_Std, x=EmBarked_Std.index, y='Fare')

fig.show()


# In[195]:


#17


# In[197]:


fig = px.bar(Pclass_GB_Mean, x=Pclass_GB_Mean.index, y='Age')

fig.show()


# In[199]:


#18


# In[201]:


import plotly.graph_objects as go


# In[202]:


new_df = df.pivot_table(index='Pclass', columns='Sex', values='Fare')

new_df


# In[203]:


fig = go.Figure(data=go.Heatmap(
        z=new_df,
        x=new_df.index,
        y=new_df.columns,
        colorscale='Picnic'))

fig.show()


# In[204]:


#19


# In[212]:


fig = px.imshow(df.corr())

fig.show()


# In[215]:


fig = go.Figure(data=go.Heatmap(
        z=df.corr(),
        x=df.corr().index,
        y=df.corr().columns,
        colorscale='Viridis'))

fig.show()


# In[216]:


#20


# In[261]:


df2 = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')

df2


# In[262]:


for col in df2.columns:
    df2[col] = df2[col].astype(str)
    
df2['text'] = df2['state'] + '<br>' +     'Exports' + df2['total exports'] + '<br>' +     'Fruits' + df2['total fruits'] + '<br>' +     'Veggies' + df2['total veggies']

df2.head()


# In[232]:


fig = go.Figure(data=go.Choropleth(
    locations=df2['code'],
    z=df2['cotton'].astype(float),
    locationmode = 'USA-states',
    text = df2['text'], 
    colorscale = 'Reds',
    colorbar_title = 'Cotton'))

fig.update_layout(
    title_text = '2011 Cotton Exports by State',
    geo_scope = 'usa')

fig.show()


# In[233]:


#21


# In[263]:


for col in df2.columns:
    df2[col] = df2[col].astype(str)
    
df2['text2'] = df2['state'] + '<br>' +     'Beef' +  df2['beef'] + ' : ' + 'Pork' + df2['pork']      #ทำให้เป็นทศนิยม 2 ตำแหน่งไม่ได้ครับ
    

df2.head()


# In[264]:


fig = go.Figure(data=go.Choropleth(
    locations=df2['code'],
    z=df2['total veggies'].astype(float),
    locationmode = 'USA-states',
    text = df2['text2'], 
    colorscale = 'Picnic',
    colorbar_title = 'Total veggies'))

fig.update_layout(
    title_text = '2011 Beef&Pork Exports by State',
    geo_scope = 'usa')

fig.show()


# In[265]:


#22


# In[279]:


df2['cotton'] = df2['cotton'].astype(float)

df2['pork'] = df2['pork'].astype(float)


# In[284]:


df2['cotton&pork'] = df2['cotton']/df2['pork']
    

df2.head()


# In[293]:


df2['code'].sample(n=10)


# In[292]:


fig = go.Figure(data=go.Choropleth(
    locations=df2['code'].sample(n=10),
    z=df2['cotton&pork'].astype(float),
    locationmode = 'USA-states',
    text = df2['total exports'],                          #อันนี้ลองทำเล่นๆครับ
    colorscale = 'Picnic',
    colorbar_title = 'Cotton&Pork'))

fig.update_layout(
    title_text = '2011 Cotton&Pork Exports by State',
    geo_scope = 'usa')

fig.show()


# In[ ]:


#อันนี้ทำจริงๆ ครับ


# In[303]:


df3 = df2.sample(n=10)

df3


# In[305]:


df3 = df3.fillna(0)

df3


# In[306]:


fig = px.bar(df3, x='state', y='total exports', 
             hover_data=['total fruits', 'total veggies'], 
             color='cotton&pork', labels={'x':'รัฐต่างๆ', 'y':'มูลค่าการส่งออกรวม'})

fig.update_layout(xaxis_tickangle=0)

fig.show()


# In[ ]:




