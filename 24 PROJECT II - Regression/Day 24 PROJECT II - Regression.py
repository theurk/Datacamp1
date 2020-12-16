#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#1 


# In[3]:


df = pd.read_csv('Train Regression.csv')
df


# In[4]:


#2


# In[5]:


df.head(10)


# In[6]:


df.tail(10)


# In[7]:


df.sample(10)


# In[8]:


#3


# In[9]:


df.info()


# In[10]:


df.describe()


# In[11]:


#4


# In[13]:


sns.pairplot(df)


# In[14]:


#5


# In[22]:


sns.distplot(df['Item_Weight'].dropna())


# In[27]:


sns.distplot(df['Item_Visibility'])


# In[28]:


sns.distplot(df['Item_MRP'])


# In[29]:


sns.distplot(df['Outlet_Establishment_Year'])


# In[30]:


sns.distplot(df['Item_Outlet_Sales'])


# In[31]:


#6


# In[33]:


fig = plt.figure(figsize=(12,8))
sns.countplot(x='Item_Type', data=df, palette='rainbow_r')
plt.xticks(rotation=45)


# In[34]:


#7 + 8


# In[36]:


df.corr()


# In[37]:


sns.heatmap(df.corr())


# In[38]:


#9


# In[39]:


plt.title('Best correlation')
plt.xlabel('Item_Outlet_Sales')
plt.ylabel('Item_MRP')
plt.scatter(df['Item_Outlet_Sales'], df['Item_MRP'])


# In[41]:


#10


# In[42]:


fig = plt.figure(figsize=(12,8))
plt.title('Worst correlation')
plt.xlabel('Item_Outlet_Sales')
plt.ylabel('Item_Visibility')
plt.scatter(df['Item_Outlet_Sales'], df['Item_Visibility'])


# In[43]:


#11


# In[44]:


df.info()


# In[45]:


df['Item_Fat_Content'].value_counts()


# In[46]:


df['Item_Fat_Content'] = df['Item_Fat_Content'].replace(['Low Fat','Regular','reg','low fat'],['LF','REG','REG','LF'])


# In[48]:


df['Item_Fat_Content'].value_counts()


# In[49]:


df['Item_Type'].value_counts()


# In[50]:


df['Outlet_Size'].value_counts()


# In[51]:


df['Outlet_Location_Type'].value_counts()


# In[52]:


df['Outlet_Type'].value_counts()


# In[54]:


#12


# In[59]:


fig = plt.figure(figsize=(12,8))
sns.countplot(x='Outlet_Type', data=df, palette='rainbow_r',hue='Item_Fat_Content')


# In[60]:


#13


# In[65]:


fig = plt.figure(figsize=(12,8))
sns.scatterplot(x='Item_MRP', y='Item_Outlet_Sales', hue='Item_Fat_Content', size='Item_Weight',
                
                palette='plasma', data=df)


# In[66]:


#14


# In[67]:


fig = plt.figure(figsize=(12,8))
sns.boxplot(x='Item_Type',y='Item_Outlet_Sales', data=df)
fig.autofmt_xdate()


# In[68]:


#15


# In[74]:


fig = plt.figure(figsize=(12,8))
sns.boxplot(x='Outlet_Size',y='Item_Outlet_Sales', data=df)


# In[75]:


#16


# In[76]:


fig = plt.figure(figsize=(12,8))
sns.boxplot(x='Outlet_Type',y='Item_Outlet_Sales', data=df)


# In[77]:


#17


# In[78]:


fig = plt.figure(figsize=(12,8))
sns.boxplot(x='Outlet_Location_Type',y='Item_Outlet_Sales', data=df)


# In[79]:


#18


# In[80]:


fig = plt.figure(figsize=(12,8))
sns.boxplot(x='Item_Fat_Content',y='Item_Outlet_Sales', data=df)


# In[81]:


#19


# In[82]:


fig = plt.figure(figsize=(12,8))
sns.boxplot(x='Item_Fat_Content',y='Item_MRP', data=df)


# In[83]:


#20


# In[84]:


fig = plt.figure(figsize=(12,8))
sns.boxplot(x='Item_Fat_Content',y='Item_Weight', data=df)


# In[85]:


#21


# In[86]:


import plotly.express as px


# In[88]:


fig = px.pie(df, values='Item_Outlet_Sales', names='Item_Type', title='Item Outlet Sales Share Per Item Type')
fig.show()


# In[89]:


#22


# In[91]:


df.head()


# In[92]:


df['Item_Identifier'].value_counts()


# In[93]:


df.groupby('Item_Identifier').mean()


# In[94]:


#23


# In[95]:


df2 = df.groupby('Item_Identifier').mean().head(10)
df2


# In[98]:


fig = px.scatter(df2, x=df2.index, y='Item_Outlet_Sales', size='Item_Weight', color='Item_Visibility')
fig.show()


# In[99]:


#24


# In[101]:


df.sample(20)


# In[102]:


df[df['Outlet_Type']=='Grocery Store']


# In[103]:


#วิธีที่ 2


# In[105]:


df.groupby('Outlet_Type').get_group('Grocery Store')


# In[106]:


#25


# In[108]:


df[(df['Item_Type']=='Soft Drinks') & (df['Outlet_Size']=='Small')]


# In[110]:


#26  bitwise operator ใช้เปรียบเทียบทีละหลายๆตัว
    #logical operator ใช้เปรียบเทียบลักษณะตัวต่อตัว


# In[112]:


#27


# In[113]:


df.info()


# In[118]:


df['Outlet_Establishment_Year'] = df['Outlet_Establishment_Year'].astype('category')


# In[119]:


df.info()


# In[120]:


#28


# In[121]:


df.isnull()


# In[125]:


fig = plt.figure(figsize=(12,8))
sns.heatmap(df.isnull(), cbar=False)


# In[126]:


#29


# In[127]:


df.info()


# In[131]:


average = df['Item_Weight'].mean()
average


# In[132]:


medium = df['Outlet_Size'].value_counts()
medium


# In[134]:


df['Item_Weight'].fillna(value=average, inplace=True)


# In[135]:


sns.heatmap(df.isnull(), cbar=False)


# In[136]:


df['Outlet_Size'].fillna(value='Medium', inplace=True)


# In[137]:


sns.heatmap(df.isnull(), cbar=False)


# In[139]:


df.info()


# In[138]:


#30


# In[140]:


df


# In[141]:


df = df.drop(['Item_Identifier','Outlet_Identifier'], axis=1)

df


# In[142]:


#31


# In[144]:


print(df['Item_Outlet_Sales'].max())
print(df['Item_Outlet_Sales'].min())


# In[145]:


#32


# In[146]:


from sklearn.model_selection import train_test_split


# In[147]:


df_real = pd.get_dummies(df, drop_first=True)
df_real


# In[148]:


#32


# In[284]:


X = df['Item_MRP']
X


# In[285]:


y = df['Item_Outlet_Sales']
y


# In[286]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[287]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[288]:


X_train = np.array(X_train).reshape(-1,1)
X_train


# In[289]:


X_train.shape


# In[290]:


X_test = np.array(X_test).reshape(-1,1)
X_test


# In[291]:


X_test.shape


# In[161]:


#33


# In[162]:


from sklearn.linear_model import LinearRegression

lm = LinearRegression()


# In[163]:


lm.fit(X_train, y_train)


# In[164]:


#34


# In[165]:


print('intercept = ', lm.intercept_)


# In[166]:


print('coefficient = ', lm.coef_)


# In[167]:


#35


# In[169]:


predicted = lm.predict(X_test)

predicted


# In[170]:


from sklearn import metrics


# In[173]:


print('MAE :', metrics.mean_absolute_error(y_test, predicted))
print('MSE :', metrics.mean_squared_error(y_test, predicted))
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, predicted)))


# In[174]:


print('R2 :', metrics.r2_score(y_test, predicted))


# In[175]:


#36


# In[179]:


fig = plt.figure(figsize=(12,8))
sns.distplot((y_test-predicted), bins=50)

#เช็คแล้ว เป็นลักษณะของ Normal distribution 
# คือ ผลต่างค่าทำนาย กับค่าจริง ได้ค่าเท่ากับ 0 เยอะ
# แปลว่า ทำนายถูกเยอะ แสดงว่าโมเดลค่อนข้างมีประสิทธิภาพ


# In[180]:


dict_compare = {'Sales' : y_test, 'Predicted' : predicted}

df_predicted = pd.DataFrame(dict_compare)
df_predicted


# In[181]:


print(df_predicted.to_string())


# In[183]:


#37


# In[186]:


fig = plt.figure(figsize=(12,8))
plt.scatter(X_test, y_test, color='blue', label='real data')
plt.plot(X_test, predicted, color='red', label='Predicted Regression Line')
plt.xlabel('Item MRP')
plt.ylabel('Item_Outlet_Sales')
plt.xlim(31, 280)
plt.ylim(0,13500)


# In[187]:


#38 + 39


# In[292]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()


# In[293]:


X = np.array(X).reshape(-1,1)
y = np.array(y).reshape(-1,1)


# In[294]:


X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y)


# In[295]:


X


# In[296]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[297]:



from sklearn.svm import SVR

regressor = SVR(kernel='rbf')


# In[298]:


regressor.fit(X_train, y_train)


# In[246]:


#40


# In[247]:


predicted_svr = regressor.predict(X_test)

predicted_svr


# In[248]:


predicted_svr = sc_y.inverse_transform(predicted_svr)

predicted_svr


# In[249]:


y_test = sc_y.inverse_transform(y_test)
y_test


# In[250]:


print('MAE :', metrics.mean_absolute_error(y_test, predicted_svr))
print('MSE :', metrics.mean_squared_error(y_test, predicted_svr))
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, predicted_svr)))


# In[251]:


print('R2 :', metrics.r2_score(y_test, predicted_svr))


# In[252]:


#41


# In[273]:


df_y_test = pd.DataFrame(data=y_test)

df_y_test


# In[274]:


fig = plt.figure(figsize=(12,8))
sns.distplot((df_y_test[0]-predicted_svr), bins=50)


# In[275]:


#42


# In[282]:


sc_x.inverse_transform(X_test)


# In[283]:


fig = plt.figure(figsize=(12,8))
plt.scatter(sc_x.inverse_transform(X_test), y_test, color='blue', label='real data')
plt.plot(sc_x.inverse_transform(X_test), predicted_svr, color='red', label='Predicted SVR Regression Line')
plt.xlabel('Item MRP')
plt.ylabel('Item_Outlet_Sales')
plt.legend()
#plt.xlim(31, 280)
#plt.ylim(0,13500)


# In[299]:


#จับคู่ X_test, y_test เพื่อทำ data visualization ของ SVR

#เนื่องจากกราฟออกมาเป็นเส้นมั่วๆ


# In[300]:


X_test


# In[301]:


y_test


# In[302]:


X_test.flatten()

#ทำให้มันกลายเป็นบรรทัดเดียวกันก่อน


# In[304]:


pair = dict(zip(X_test.flatten(), y_test.flatten()))

pair


# In[307]:


pair = dict(sorted(pair.items()))

pair


# In[309]:


list(pair.keys())


# In[310]:


X_test = np.array(list(pair.keys()))

X_test


# In[311]:


X_test = np.array(list(pair.keys())).reshape(-1,1)

X_test


# In[312]:


y_test = np.array(list(pair.values())).reshape(-1,1)

y_test


# In[313]:


predicted_svr = regressor.predict(X_test)

predicted_svr


# In[314]:


predicted_svr = sc_y.inverse_transform(predicted_svr)

predicted_svr


# In[315]:


y_test = sc_y.inverse_transform(y_test)
y_test


# In[316]:


print('MAE :', metrics.mean_absolute_error(y_test, predicted_svr))
print('MSE :', metrics.mean_squared_error(y_test, predicted_svr))
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, predicted_svr)))


# In[317]:


print('R2 :', metrics.r2_score(y_test, predicted_svr))


# In[318]:


sc_x.inverse_transform(X_test)


# In[320]:


fig = plt.figure(figsize=(12,8))
plt.scatter(sc_x.inverse_transform(X_test), y_test, color='blue', label='real data')
plt.plot(sc_x.inverse_transform(X_test), predicted_svr, color='red', label='Predicted SVR Regression Line')
plt.xlabel('Item MRP')
plt.ylabel('Item_Outlet_Sales')
plt.legend()
#plt.xlim(31, 280)
#plt.ylim(0,13500)

#ได้เส้นที่สวยงามแล้ววว  สังเกตจะไม่ใช้เส้นตรง เส้นมันจะโค้งหน่อยๆ


# ###################

# In[322]:


#43


# In[392]:


df


# In[393]:


df_real


# In[394]:


sc_x = StandardScaler()
sc_y = StandardScaler()


# In[395]:


X = df_real.drop(['Item_Outlet_Sales'], axis=1)
y = np.array(df_real['Item_Outlet_Sales']).reshape(-1,1)


# In[396]:


X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y)


# In[397]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[398]:


lm = LinearRegression()


# In[399]:


lm.fit(X_train, y_train)


# In[400]:


#44


# In[401]:


print('intercept = ', lm.intercept_)
print('coefficient = ', lm.coef_)


# In[403]:


len(lm.coef_[0])


# In[404]:


#45


# In[407]:


predicted = sc_y.inverse_transform(lm.predict(X_test))

predicted


# In[408]:


y_real = sc_y.inverse_transform(y_test)
y_real


# In[409]:


print('MAE :', metrics.mean_absolute_error(y_real, predicted))
print('MSE :', metrics.mean_squared_error(y_real, predicted))
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_real, predicted)))
print('R2 :', metrics.r2_score(y_real, predicted))

#คะแนนแบบ all features ได้สูงกว่า แบบเลือกแค่บาง feature


# In[410]:


#46


# In[411]:


fig = plt.figure(figsize=(12,8))
sns.distplot((y_real-predicted), bins=50)


# In[412]:


#47


# In[413]:


dict_compare = {'Sales' : y_real.flatten(), 'Predicted' : predicted.flatten()}

df_predicted = pd.DataFrame(dict_compare)
df_predicted


# In[414]:


print(df_predicted.to_string())


# In[415]:


#48


# In[416]:


df_predicted.corr()


# In[417]:


# linear regression ไม่จำเป็นต้องทำ standardize


# In[418]:


#49


# In[419]:


regressor = SVR(kernel='rbf')
regressor.fit(X_train, y_train)


# In[420]:


predicted_svr = regressor.predict(X_test)

predicted_svr


# In[421]:


predicted_svr = sc_y.inverse_transform(predicted_svr)

predicted_svr


# In[422]:


y_real_svr = sc_y.inverse_transform(y_test)
y_real_svr


# In[423]:


#50


# In[424]:


print('MAE :', metrics.mean_absolute_error(y_real_svr, predicted_svr))
print('MSE :', metrics.mean_squared_error(y_real_svr, predicted_svr))
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_real_svr, predicted_svr)))
print('R2 :', metrics.r2_score(y_real_svr, predicted_svr))


# In[425]:


#51


# In[426]:


df_y_real_svr = pd.DataFrame(data=y_real_svr)

df_y_real_svr


# In[427]:


fig = plt.figure(figsize=(12,8))
sns.distplot((df_y_real_svr[0]-predicted_svr), bins=50)


# In[428]:


#52


# In[429]:


dict_compare = {'Sales' : y_real_svr.flatten(), 'Predicted SVR' : predicted_svr.flatten()}

df_predicted_svr = pd.DataFrame(dict_compare)
df_predicted_svr


# In[430]:


print(df_predicted_svr.to_string())


# In[431]:


#53


# In[432]:


df_predicted_svr.corr()


# In[433]:


# SVR ควรทำ Standardize


# In[434]:


#54


# In[435]:


from sklearn.tree import DecisionTreeRegressor


# In[454]:


dt_regressor = DecisionTreeRegressor()


# In[455]:


dt_regressor.fit(X_train, y_train)


# In[456]:


predicted_dt_r = dt_regressor.predict(X_test)

predicted_dt_r


# In[457]:


predicted_dt_r = sc_y.inverse_transform(predicted_dt_r)

predicted_dt_r


# In[458]:


#55


# In[459]:


y_real_dt_r = sc_y.inverse_transform(y_test)
y_real_dt_r


# In[461]:


print('MAE :', metrics.mean_absolute_error(y_real_dt_r, predicted_dt_r))
print('MSE :', metrics.mean_squared_error(y_real_dt_r, predicted_dt_r))
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_real_dt_r, predicted_dt_r)))
print('R2 :', metrics.r2_score(y_real_dt_r, predicted_dt_r))


# In[466]:


#56


# In[469]:


df_y_real_dt_r = pd.DataFrame(data=y_real_dt_r)

df_y_real_dt_r


# In[471]:


fig = plt.figure(figsize=(12,8))
sns.distplot((df_y_real_dt_r[0]-predicted_dt_r), bins=50)


# In[472]:


#57


# In[473]:


dict_compare = {'Sales' : y_real_dt_r.flatten(), 'Predicted SVR' : predicted_dt_r.flatten()}

df_predicted_dt_r = pd.DataFrame(dict_compare)
df_predicted_dt_r


# In[474]:


#58


# In[475]:


df_predicted_dt_r.corr()


# In[476]:


# Decision Tree ไม่จำเป็นต้อง Standardize


# In[477]:


#59


# In[483]:


from sklearn.ensemble import RandomForestRegressor


# In[484]:


rf_r = RandomForestRegressor()


# In[485]:


rf_r.fit(X_train, y_train)


# In[488]:


predicted_rf_r = rf_r.predict(X_test)

predicted_rf_r


# In[489]:


predicted_rf_r = sc_y.inverse_transform(predicted_rf_r)

predicted_rf_r


# In[491]:


#60


# In[492]:


y_real_rf_r = sc_y.inverse_transform(y_test)
y_real_rf_r


# In[493]:


print('MAE :', metrics.mean_absolute_error(y_real_rf_r, predicted_rf_r))
print('MSE :', metrics.mean_squared_error(y_real_rf_r, predicted_rf_r))
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_real_rf_r, predicted_rf_r)))
print('R2 :', metrics.r2_score(y_real_rf_r, predicted_rf_r))


# In[494]:


#61


# In[495]:


df_y_real_rf_r = pd.DataFrame(data=y_real_rf_r)

df_y_real_rf_r


# In[496]:


fig = plt.figure(figsize=(12,8))
sns.distplot((df_y_real_rf_r[0]-predicted_rf_r), bins=50)


# In[497]:


#62


# In[498]:


dict_compare = {'Sales' : y_real_rf_r.flatten(), 'Predicted SVR' : predicted_rf_r.flatten()}

df_predicted_rf_r = pd.DataFrame(dict_compare)
df_predicted_rf_r


# In[499]:


#63


# In[500]:


df_predicted_rf_r.corr()


# In[502]:


# Random Forest ไม่จำเป็นต้องทำ Standardardize


# # ข้อ 64 - 83 ทำไปแล้ว

# In[507]:


#84


# In[550]:


X2 = ['Linear Regression', 'Support Vector Machine', 'Decision Tree', 'Random Forest']
y2 = [1152,1123,1606,1165]


# In[551]:


fig = plt.figure(figsize=(10,8))
sns.barplot(x=X2,y=y2)
plt.title('RMSE of four models (less is better)')
plt.xlabel('Models Title')
plt.ylabel('RMSE Score')


# In[511]:


#85


# In[552]:


X2 = ['Linear Regression', 'Support Vector Machine', 'Decision Tree', 'Random Forest']
y2 = [0.54,0.57,0.12,0.53]


# In[553]:


fig = plt.figure(figsize=(10,8))
sns.barplot(x=X2,y=y2)
plt.title('R2 of four models (More is better)')
plt.xlabel('Models Title')
plt.ylabel('R2 Score')


# # ทำย้อนหลังข้อ 43 - 63 แบบไม่ทำ Standardize

# In[515]:


X3 = df_real.drop(['Item_Outlet_Sales'], axis=1)
y3 = df_real['Item_Outlet_Sales']


# In[516]:


X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size=0.2, random_state=100)


# In[517]:


X_train


# # Linear

# In[519]:


lm2 = LinearRegression()


# In[520]:


lm2.fit(X_train, y_train)


# In[521]:


print('intercept = ', lm2.intercept_)
print('coefficient = ', lm2.coef_)


# In[522]:


predicted = lm2.predict(X_test)

predicted


# In[524]:


# Linear No stadardize

print('MAE :', metrics.mean_absolute_error(y_test, predicted))
print('MSE :', metrics.mean_squared_error(y_test, predicted))
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, predicted)))
print('R2 :', metrics.r2_score(y_test, predicted))


# In[525]:


fig = plt.figure(figsize=(12,8))
sns.distplot((y_test-predicted), bins=50)


# In[527]:


dict_compare = {'Sales' : y_test, 'Predicted' : predicted}

df_predicted = pd.DataFrame(dict_compare)
df_predicted


# In[528]:


df_predicted.corr()


# # SVR

# In[529]:


regressor2 = SVR(kernel='rbf')
regressor2.fit(X_train, y_train)


# In[531]:


predicted_svr = regressor2.predict(X_test)

predicted_svr


# In[532]:


# SVR No stadardize

print('MAE :', metrics.mean_absolute_error(y_test, predicted_svr))
print('MSE :', metrics.mean_squared_error(y_test, predicted_svr))
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, predicted_svr)))
print('R2 :', metrics.r2_score(y_test, predicted_svr))


# In[533]:


fig = plt.figure(figsize=(12,8))
sns.distplot((y_test-predicted_svr), bins=50)


# In[534]:


dict_compare = {'Sales' : y_test, 'Predicted' : predicted_svr}

df_predicted_svr = pd.DataFrame(dict_compare)
df_predicted_svr


# In[535]:


df_predicted_svr.corr()


# # Decision Tree

# In[536]:


dt_regressor2 = DecisionTreeRegressor()
dt_regressor2.fit(X_train, y_train)


# In[537]:


predicted_dt = dt_regressor2.predict(X_test)

predicted_dt


# In[538]:


# Decision Tree No stadardize

print('MAE :', metrics.mean_absolute_error(y_test, predicted_dt))
print('MSE :', metrics.mean_squared_error(y_test, predicted_dt))
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, predicted_dt)))
print('R2 :', metrics.r2_score(y_test, predicted_dt))


# In[539]:


fig = plt.figure(figsize=(12,8))
sns.distplot((y_test-predicted_dt), bins=50)


# In[540]:


dict_compare = {'Sales' : y_test, 'Predicted dt' : predicted_dt}

df_predicted_dt = pd.DataFrame(dict_compare)
df_predicted_dt


# In[541]:


df_predicted_dt.corr()


# # Random Forest

# In[542]:


rf_r2 = RandomForestRegressor()
rf_r2.fit(X_train, y_train)


# In[543]:


predicted_rf = rf_r2.predict(X_test)

predicted_rf


# In[544]:


# Random Forest No stadardize

print('MAE :', metrics.mean_absolute_error(y_test, predicted_rf))
print('MSE :', metrics.mean_squared_error(y_test, predicted_rf))
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, predicted_rf)))
print('R2 :', metrics.r2_score(y_test, predicted_rf))


# In[545]:


fig = plt.figure(figsize=(12,8))
sns.distplot((y_test-predicted_rf), bins=50)


# In[546]:


dict_compare = {'Sales' : y_test, 'Predicted rf' : predicted_rf}

df_predicted_rf = pd.DataFrame(dict_compare)
df_predicted_rf


# In[547]:


df_predicted_rf.corr()


# In[ ]:




