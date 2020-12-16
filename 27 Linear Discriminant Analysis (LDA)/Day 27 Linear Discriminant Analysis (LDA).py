#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # 1

# In[2]:


df = pd.read_csv('Wine_completed.csv')
df


# # 2

# In[3]:


df.head(10)


# In[4]:


df.tail(10)


# In[5]:


df.sample(10)


# # 3 + 4

# In[6]:


df.info()


# In[7]:


df.describe()


# # 5

# In[8]:


sns.pairplot(df)


# # 6

# In[9]:


sns.distplot(df['Alcohol'])


# In[10]:


sns.distplot(df['Malic acid'])


# In[11]:


sns.distplot(df['Ash'])


# In[12]:


sns.distplot(df['Alcalinity of ash'])


# In[13]:


sns.distplot(df['Magnesium'])


# In[14]:


sns.distplot(df['Flavanoids'])


# # 7

# In[15]:


df.corr()


# In[16]:


fig = plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linecolor='white', linewidth=2)


# # 8

# In[17]:


plt.title('Best correlation')
plt.xlabel('Flavanoids')
plt.ylabel('OD280/OD315 of diluted wines')
plt.scatter(df['Flavanoids'], df['OD280/OD315 of diluted wines'])


# # 9

# In[18]:


plt.title('Worst correlation')
plt.xlabel('Malic acid')
plt.ylabel('Hue')
plt.scatter(df['Malic acid'], df['Hue'])


# # 10

# In[19]:


fig = plt.figure(figsize=(12,8))
plt.hist(df['Malic acid'],bins=10)
plt.show()


# In[20]:


fig = plt.figure(figsize=(12,8))
plt.hist(df['Flavanoids'],bins=10)
plt.show()


# # 11

# In[21]:


fig = plt.figure(figsize=(12,8))
sns.boxplot(x='Class',y='Malic acid', data=df)


# In[22]:


fig = plt.figure(figsize=(12,8))
sns.boxplot(x='Class',y='Flavanoids', data=df)


# In[23]:


fig = plt.figure(figsize=(12,8))
sns.boxplot(x='Class',y='Magnesium', data=df)


# # 12

# In[24]:


#ไม่ได้จัดการ Outlier เนื่องจากขาดความรู้ในเรื่องสารเคมีต่างๆครับ


# # 13

# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


df


# In[27]:


X = df.drop(['Class'], axis = 1)
X


# In[28]:


y = df['Class']
y


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


# In[30]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # 14

# In[31]:


from sklearn.preprocessing import StandardScaler


# In[32]:


sc = StandardScaler()


# In[33]:


X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[34]:


X_train


# # 15  

# In[35]:


# เลือก Baseline 3 ชนิด

# 1. Logistic Regression
# 2. Naive Bayes
# 3. Random Forest


# # 16   Baseline 1 Logistic Regression

# In[36]:


from sklearn.linear_model import LogisticRegression


# In[37]:


logistic_classifier = LogisticRegression()
logistic_classifier.fit(X_train, y_train)


# In[38]:


logistic_classifier_predicted = logistic_classifier.predict(X_test)


# # 17

# In[39]:


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


# In[40]:


confusion_matrix(y_test,logistic_classifier_predicted)


# In[41]:


print('Logistic Regression Result(micro)')

print('Accuracy = ', accuracy_score(y_test,logistic_classifier_predicted))
print('F1-score = ', f1_score(y_test,logistic_classifier_predicted, average='micro'))
print('Precision = ', precision_score(y_test,logistic_classifier_predicted, average='micro'))
print('Recall = ', recall_score(y_test,logistic_classifier_predicted, average='micro'))


# In[42]:


print('Logistic Regression Result(macro)')

print('Accuracy = ', accuracy_score(y_test,logistic_classifier_predicted))
print('F1-score = ', f1_score(y_test,logistic_classifier_predicted, average='macro'))
print('Precision = ', precision_score(y_test,logistic_classifier_predicted, average='macro'))
print('Recall = ', recall_score(y_test,logistic_classifier_predicted, average='macro'))


# # Baseline 2 Naive Bayes

# In[43]:


from sklearn.naive_bayes import GaussianNB


# In[44]:


nb = GaussianNB()
nb.fit(X_train, y_train)


# In[45]:


nb_predicted = nb.predict(X_test)


# In[46]:


confusion_matrix(y_test,nb_predicted)


# In[47]:


print('Naive Bayes Result(micro)')

print('Accuracy = ', accuracy_score(y_test,nb_predicted))
print('F1-score = ', f1_score(y_test,nb_predicted, average='micro'))
print('Precision = ', precision_score(y_test,nb_predicted, average='micro'))
print('Recall = ', recall_score(y_test,nb_predicted, average='micro'))


# In[48]:


print('Naive Bayes Result(macro)')

print('Accuracy = ', accuracy_score(y_test,nb_predicted))
print('F1-score = ', f1_score(y_test,nb_predicted, average='macro'))
print('Precision = ', precision_score(y_test,nb_predicted, average='macro'))
print('Recall = ', recall_score(y_test,nb_predicted, average='macro'))


# # Baseline 3 Random Forest

# In[49]:


from sklearn.ensemble import RandomForestClassifier


# In[50]:


rf = RandomForestClassifier()
rf.fit(X_train, y_train)


# In[51]:


rf_predicted = rf.predict(X_test)


# In[52]:


confusion_matrix(y_test,rf_predicted)


# In[53]:


print('Random Forest Result(micro)')

print('Accuracy = ', accuracy_score(y_test,rf_predicted))
print('F1-score = ', f1_score(y_test,rf_predicted, average='micro'))
print('Precision = ', precision_score(y_test,rf_predicted, average='micro'))
print('Recall = ', recall_score(y_test,rf_predicted, average='micro'))


# In[54]:


print('Random Forest Result(macro)')

print('Accuracy = ', accuracy_score(y_test,rf_predicted))
print('F1-score = ', f1_score(y_test,rf_predicted, average='macro'))
print('Precision = ', precision_score(y_test,rf_predicted, average='macro'))
print('Recall = ', recall_score(y_test,rf_predicted, average='macro'))


# # 18  ทำ Grid Search

# In[55]:


from sklearn.model_selection import GridSearchCV


# # Logistic Regression + GridSearch

# In[56]:


param_grid = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}


# In[57]:


grid_search =  GridSearchCV(LogisticRegression(), param_grid, verbose=3)


# In[58]:


grid_search.fit(X_train, y_train)


# In[59]:


grid_search.best_params_


# In[60]:


grid_search.best_estimator_


# In[61]:


logistic_classifier_grid_predicted = grid_search.predict(X_test)


# # 19 วัดผล หลังทำ Parameter tuning

# In[62]:


confusion_matrix(y_test,logistic_classifier_grid_predicted)


# In[63]:


print('Logistic Regression + GridSearch Result(micro)')

print('Accuracy = ', accuracy_score(y_test,logistic_classifier_grid_predicted))
print('F1-score = ', f1_score(y_test,logistic_classifier_grid_predicted, average='micro'))
print('Precision = ', precision_score(y_test,logistic_classifier_grid_predicted, average='micro'))
print('Recall = ', recall_score(y_test,logistic_classifier_grid_predicted, average='micro'))


# In[64]:


print('Logistic Regression + GridSearch Result(macro)')

print('Accuracy = ', accuracy_score(y_test,logistic_classifier_grid_predicted))
print('F1-score = ', f1_score(y_test,logistic_classifier_grid_predicted, average='macro'))
print('Precision = ', precision_score(y_test,logistic_classifier_grid_predicted, average='macro'))
print('Recall = ', recall_score(y_test,logistic_classifier_grid_predicted, average='macro'))


# # Naive Bayes + GridSearch

# In[65]:


param_grid = {'var_smoothing' : [1e-09,2e-05,1e-5,1e-7,3e-1]}


# In[66]:


grid_search =  GridSearchCV(GaussianNB(), param_grid, verbose=3)


# In[67]:


grid_search.fit(X_train, y_train)


# In[68]:


grid_search.best_params_


# In[69]:


grid_search.best_estimator_


# In[70]:


nb_grid_predicted = grid_search.predict(X_test)


# In[71]:


confusion_matrix(y_test,nb_grid_predicted)


# In[72]:


print('Naive Bayes + GridSearch Result(micro)')

print('Accuracy = ', accuracy_score(y_test,nb_grid_predicted))
print('F1-score = ', f1_score(y_test,nb_grid_predicted, average='micro'))
print('Precision = ', precision_score(y_test,nb_grid_predicted, average='micro'))
print('Recall = ', recall_score(y_test,nb_grid_predicted, average='micro'))


# In[73]:


print('Naive Bayes + GridSearch Result(macro)')

print('Accuracy = ', accuracy_score(y_test,nb_grid_predicted))
print('F1-score = ', f1_score(y_test,nb_grid_predicted, average='macro'))
print('Precision = ', precision_score(y_test,nb_grid_predicted, average='macro'))
print('Recall = ', recall_score(y_test,nb_grid_predicted, average='macro'))


# # Random Forest + GridSearch

# In[74]:


param_grid = {'max_depth':[4,8,16,None],
              'max_features':[4,8],
              'n_estimators':[50,100,200,500],
              'min_samples_split':[3,5,6,7]}


# In[75]:


grid_search =  GridSearchCV(RandomForestClassifier(), param_grid, verbose=3)


# In[76]:


grid_search.fit(X_train, y_train)


# In[77]:


grid_search.best_params_


# In[78]:


grid_search.best_estimator_


# In[79]:


rf_grid_predicted = grid_search.predict(X_test)


# In[80]:


print('Naive Bayes + GridSearch Result(micro)')

print('Accuracy = ', accuracy_score(y_test,rf_grid_predicted))
print('F1-score = ', f1_score(y_test,rf_grid_predicted, average='micro'))
print('Precision = ', precision_score(y_test,rf_grid_predicted, average='micro'))
print('Recall = ', recall_score(y_test,rf_grid_predicted, average='micro'))


# In[81]:


print('Naive Bayes + GridSearch Result(macro)')

print('Accuracy = ', accuracy_score(y_test,rf_grid_predicted))
print('F1-score = ', f1_score(y_test,rf_grid_predicted, average='macro'))
print('Precision = ', precision_score(y_test,rf_grid_predicted, average='macro'))
print('Recall = ', recall_score(y_test,rf_grid_predicted, average='macro'))


# # 20 LDA

# In[82]:


sns.distplot(df['Malic acid'])


# In[83]:


sns.distplot(df['Alcohol'])


# In[84]:


#การทำ LDA จริงๆควรมีการกระจายตัวแบบปกติ ถึงจะได้ผลลัพธ์ที่ดีกว่า


# In[85]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In[86]:


lda = LDA(n_components=2)


# In[87]:


X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)


# In[88]:


X_train_lda


# In[89]:


print(X_train_lda.shape)
print(X_test_lda.shape)


# # 21

# In[91]:


X_train_lda


# In[92]:


X_train_lda[:,0]


# In[93]:


fig = plt.figure(figsize=(12,8))
plt.scatter(X_train_lda[:,0],X_train_lda[:,1],c=y_train, cmap='coolwarm')
plt.xlabel('LDA 1')
plt.ylabel('LDA 2')
plt.grid()


# # 22 เทรนโมเดล + LDA

# # Logistic Regression

# In[94]:


logistic_classifier_lda = LogisticRegression()
logistic_classifier_lda.fit(X_train_lda, y_train)


# In[95]:


logistic_classifier_predicted_lda = logistic_classifier_lda.predict(X_test_lda)


# # 23 วัดผล

# In[96]:


confusion_matrix(y_test,logistic_classifier_predicted_lda)


# In[97]:


print('Logistic Regression Result(micro) + LDA')

print('Accuracy = ', accuracy_score(y_test,logistic_classifier_predicted_lda))
print('F1-score = ', f1_score(y_test,logistic_classifier_predicted_lda, average='micro'))
print('Precision = ', precision_score(y_test,logistic_classifier_predicted_lda, average='micro'))
print('Recall = ', recall_score(y_test,logistic_classifier_predicted_lda, average='micro'))


# In[99]:


print('Logistic Regression Result(macro) + LDA')

print('Accuracy = ', accuracy_score(y_test,logistic_classifier_predicted_lda))
print('F1-score = ', f1_score(y_test,logistic_classifier_predicted_lda, average='macro'))
print('Precision = ', precision_score(y_test,logistic_classifier_predicted_lda, average='macro'))
print('Recall = ', recall_score(y_test,logistic_classifier_predicted_lda, average='macro'))


# # Naive Bayes

# In[100]:


nb_lda = GaussianNB()
nb_lda.fit(X_train_lda, y_train)


# In[102]:


nb_predicted_lda = nb_lda.predict(X_test_lda)


# In[103]:


confusion_matrix(y_test,nb_predicted_lda)


# In[104]:


print('Naive Bayes Result(micro) + LDA')

print('Accuracy = ', accuracy_score(y_test,nb_predicted_lda))
print('F1-score = ', f1_score(y_test,nb_predicted_lda, average='micro'))
print('Precision = ', precision_score(y_test,nb_predicted_lda, average='micro'))
print('Recall = ', recall_score(y_test,nb_predicted_lda, average='micro'))


# In[105]:


print('Naive Bayes Result(macro) + LDA')

print('Accuracy = ', accuracy_score(y_test,nb_predicted_lda))
print('F1-score = ', f1_score(y_test,nb_predicted_lda, average='macro'))
print('Precision = ', precision_score(y_test,nb_predicted_lda, average='macro'))
print('Recall = ', recall_score(y_test,nb_predicted_lda, average='macro'))


# # Random Forest

# In[106]:


rf_lda = RandomForestClassifier()
rf_lda.fit(X_train_lda, y_train)


# In[107]:


rf_predicted_lda = rf_lda.predict(X_test_lda)


# In[108]:


confusion_matrix(y_test,rf_predicted_lda)


# In[109]:


print('Random Forest Result(micro) + LDA')

print('Accuracy = ', accuracy_score(y_test,rf_predicted_lda))
print('F1-score = ', f1_score(y_test,rf_predicted_lda, average='micro'))
print('Precision = ', precision_score(y_test,rf_predicted_lda, average='micro'))
print('Recall = ', recall_score(y_test,rf_predicted_lda, average='micro'))


# In[110]:


print('Random Forest Result(macro) + LDA')

print('Accuracy = ', accuracy_score(y_test,rf_predicted_lda))
print('F1-score = ', f1_score(y_test,rf_predicted_lda, average='macro'))
print('Precision = ', precision_score(y_test,rf_predicted_lda, average='macro'))
print('Recall = ', recall_score(y_test,rf_predicted_lda, average='macro'))


# # 24  parameter tuning กับ โมเดลใหม่(LDA)

# # Logistic Regression(LDA) + GridSearch

# In[111]:


param_grid = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}


# In[112]:


grid_search =  GridSearchCV(LogisticRegression(), param_grid, verbose=3)


# In[113]:


grid_search.fit(X_train_lda, y_train)


# In[114]:


grid_search.best_params_


# In[115]:


grid_search.best_estimator_


# In[116]:


logistic_classifier_grid_predicted_lda = grid_search.predict(X_test_lda)


# # 25 วัดผล

# In[117]:


confusion_matrix(y_test,logistic_classifier_grid_predicted_lda)


# In[118]:


print('Logistic Regression + GridSearch Result(micro) + LDA')

print('Accuracy = ', accuracy_score(y_test,logistic_classifier_grid_predicted_lda))
print('F1-score = ', f1_score(y_test,logistic_classifier_grid_predicted_lda, average='micro'))
print('Precision = ', precision_score(y_test,logistic_classifier_grid_predicted_lda, average='micro'))
print('Recall = ', recall_score(y_test,logistic_classifier_grid_predicted_lda, average='micro'))


# In[119]:


print('Logistic Regression + GridSearch Result(macro) + LDA')

print('Accuracy = ', accuracy_score(y_test,logistic_classifier_grid_predicted_lda))
print('F1-score = ', f1_score(y_test,logistic_classifier_grid_predicted_lda, average='macro'))
print('Precision = ', precision_score(y_test,logistic_classifier_grid_predicted_lda, average='macro'))
print('Recall = ', recall_score(y_test,logistic_classifier_grid_predicted_lda, average='macro'))


# # Naive Bayes(LDA) + GridSearch

# In[120]:


param_grid = {'var_smoothing' : [1e-09,2e-05,1e-5,1e-7,3e-1]}


# In[121]:


grid_search =  GridSearchCV(GaussianNB(), param_grid, verbose=3)


# In[122]:


grid_search.fit(X_train_lda, y_train)


# In[123]:


grid_search.best_params_


# In[124]:


grid_search.best_estimator_


# In[125]:


nb_grid_predicted_lda = grid_search.predict(X_test_lda)


# In[126]:


confusion_matrix(y_test,nb_grid_predicted_lda)


# In[127]:


print('Naive Bayes + GridSearch Result(micro) + LDA')

print('Accuracy = ', accuracy_score(y_test,nb_grid_predicted_lda))
print('F1-score = ', f1_score(y_test,nb_grid_predicted_lda, average='micro'))
print('Precision = ', precision_score(y_test,nb_grid_predicted_lda, average='micro'))
print('Recall = ', recall_score(y_test,nb_grid_predicted_lda, average='micro'))


# In[128]:


print('Naive Bayes + GridSearch Result(macro) + LDA')

print('Accuracy = ', accuracy_score(y_test,nb_grid_predicted_lda))
print('F1-score = ', f1_score(y_test,nb_grid_predicted_lda, average='macro'))
print('Precision = ', precision_score(y_test,nb_grid_predicted_lda, average='macro'))
print('Recall = ', recall_score(y_test,nb_grid_predicted_lda, average='macro'))


# # Random Forest(LDA) + GridSearch

# In[129]:


param_grid = {'max_depth':[4,8,16,None],
              'max_features':[2],
              'n_estimators':[50,100,200,500],
              'min_samples_split':[3,5,6,7]}


# In[130]:


grid_search =  GridSearchCV(RandomForestClassifier(), param_grid, verbose=3)


# In[131]:


grid_search.fit(X_train_lda, y_train)


# In[132]:


grid_search.best_params_


# In[133]:


grid_search.best_estimator_


# In[134]:


rf_grid_predicted_lda = grid_search.predict(X_test_lda)


# In[135]:


confusion_matrix(y_test,rf_grid_predicted_lda)


# In[136]:


print('RandomForest + GridSearch Result(micro) + LDA')

print('Accuracy = ', accuracy_score(y_test,rf_grid_predicted_lda))
print('F1-score = ', f1_score(y_test,rf_grid_predicted_lda, average='micro'))
print('Precision = ', precision_score(y_test,rf_grid_predicted_lda, average='micro'))
print('Recall = ', recall_score(y_test,rf_grid_predicted_lda, average='micro'))


# In[137]:


print('RandomForest + GridSearch Result(macro) + LDA')

print('Accuracy = ', accuracy_score(y_test,rf_grid_predicted_lda))
print('F1-score = ', f1_score(y_test,rf_grid_predicted_lda, average='macro'))
print('Precision = ', precision_score(y_test,rf_grid_predicted_lda, average='macro'))
print('Recall = ', recall_score(y_test,rf_grid_predicted_lda, average='macro'))


# # 26

# In[138]:


import plotly.express as px


# In[141]:


rows = ['LR','NB','RF','LR_HT','NB_HT','RF_HT','LR_LDA','NB_LDA','RF_LDA','LR_LDA_HT','NB_LDA_HT','RF_LDA_HT']
columns = ['ACC','F1 Score','Precision','Recall']

values = [[accuracy_score(y_test, logistic_classifier_predicted),f1_score(y_test, logistic_classifier_predicted, average='micro'),precision_score(y_test, logistic_classifier_predicted, average='micro'), recall_score(y_test, logistic_classifier_predicted, average='micro')],
         [accuracy_score(y_test, nb_predicted),f1_score(y_test, nb_predicted, average='micro'),precision_score(y_test, nb_predicted, average='micro'), recall_score(y_test, nb_predicted, average='micro')],
         [accuracy_score(y_test, rf_predicted),f1_score(y_test, rf_predicted, average='micro'),precision_score(y_test, rf_predicted, average='micro'), recall_score(y_test, rf_predicted, average='micro')],
         [accuracy_score(y_test, logistic_classifier_grid_predicted),f1_score(y_test, logistic_classifier_grid_predicted, average='micro'),precision_score(y_test, logistic_classifier_grid_predicted, average='micro'), recall_score(y_test, logistic_classifier_grid_predicted, average='micro')],
         [accuracy_score(y_test, nb_grid_predicted),f1_score(y_test, nb_grid_predicted, average='micro'),precision_score(y_test, nb_grid_predicted, average='micro'), recall_score(y_test, nb_grid_predicted, average='micro')],
         [accuracy_score(y_test, rf_grid_predicted),f1_score(y_test, rf_grid_predicted, average='micro'),precision_score(y_test, rf_grid_predicted, average='micro'), recall_score(y_test, rf_grid_predicted, average='micro')],
         [accuracy_score(y_test, logistic_classifier_predicted_lda),f1_score(y_test, logistic_classifier_predicted_lda, average='micro'),precision_score(y_test, logistic_classifier_predicted_lda, average='micro'), recall_score(y_test, logistic_classifier_predicted_lda, average='micro')],
         [accuracy_score(y_test, nb_predicted_lda),f1_score(y_test, nb_predicted_lda, average='micro'),precision_score(y_test, nb_predicted_lda, average='micro'), recall_score(y_test, nb_predicted_lda, average='micro')],
         [accuracy_score(y_test, rf_predicted_lda),f1_score(y_test, rf_predicted_lda, average='micro'),precision_score(y_test, rf_predicted_lda, average='micro'), recall_score(y_test, rf_predicted_lda, average='micro')],
         [accuracy_score(y_test, logistic_classifier_grid_predicted_lda),f1_score(y_test, logistic_classifier_grid_predicted_lda, average='micro'),precision_score(y_test, logistic_classifier_grid_predicted_lda, average='micro'), recall_score(y_test, logistic_classifier_grid_predicted_lda, average='micro')],
         [accuracy_score(y_test, nb_grid_predicted_lda),f1_score(y_test, nb_grid_predicted_lda, average='micro'),precision_score(y_test, nb_grid_predicted_lda, average='micro'), recall_score(y_test, nb_grid_predicted_lda, average='micro')],
         [accuracy_score(y_test, rf_grid_predicted_lda),f1_score(y_test, rf_grid_predicted_lda, average='micro'),precision_score(y_test, rf_grid_predicted_lda, average='micro'), recall_score(y_test, rf_grid_predicted_lda, average='micro')]]


# In[142]:


df_model_Full_compare = pd.DataFrame(values,rows,columns)

df_model_Full_compare


# In[143]:


fig = px.bar(df_model_Full_compare, y='ACC', x=df_model_Full_compare.index, title='ACC Comparison')
fig.show()


# # 27

# In[144]:


rows = ['LR','NB','RF','LR_HT','NB_HT','RF_HT','LR_LDA','NB_LDA','RF_LDA','LR_LDA_HT','NB_LDA_HT','RF_LDA_HT']
columns = ['ACC','F1 Score micro','F1 Score macro','Precision micro','Precision macro','Recall micro','Recall macro']

values = [[accuracy_score(y_test, logistic_classifier_predicted),f1_score(y_test, logistic_classifier_predicted, average='micro'),f1_score(y_test, logistic_classifier_predicted, average='macro'),precision_score(y_test, logistic_classifier_predicted, average='micro'),precision_score(y_test, logistic_classifier_predicted, average='macro'), recall_score(y_test, logistic_classifier_predicted, average='micro'),recall_score(y_test, logistic_classifier_predicted, average='macro')],
         [accuracy_score(y_test, nb_predicted),f1_score(y_test, nb_predicted, average='micro'),f1_score(y_test, nb_predicted, average='macro'),precision_score(y_test, nb_predicted, average='micro'),precision_score(y_test, nb_predicted, average='macro'), recall_score(y_test, nb_predicted, average='micro'),recall_score(y_test, nb_predicted, average='macro')],
         [accuracy_score(y_test, rf_predicted),f1_score(y_test, rf_predicted, average='micro'),f1_score(y_test, rf_predicted, average='macro'),precision_score(y_test, rf_predicted, average='micro'),precision_score(y_test, rf_predicted, average='macro'), recall_score(y_test, rf_predicted, average='micro'),recall_score(y_test, rf_predicted, average='macro')],
         [accuracy_score(y_test, logistic_classifier_grid_predicted),f1_score(y_test, logistic_classifier_grid_predicted, average='micro'),f1_score(y_test, logistic_classifier_grid_predicted, average='macro'),precision_score(y_test, logistic_classifier_grid_predicted, average='micro'),precision_score(y_test, logistic_classifier_grid_predicted, average='macro'), recall_score(y_test, logistic_classifier_grid_predicted, average='micro'),recall_score(y_test, logistic_classifier_grid_predicted, average='macro')],
         [accuracy_score(y_test, nb_grid_predicted),f1_score(y_test, nb_grid_predicted, average='micro'),f1_score(y_test, nb_grid_predicted, average='macro'),precision_score(y_test, nb_grid_predicted, average='micro'), precision_score(y_test, nb_grid_predicted, average='macro'),recall_score(y_test, nb_grid_predicted, average='micro'),recall_score(y_test, nb_grid_predicted, average='macro')],
         [accuracy_score(y_test, rf_grid_predicted),f1_score(y_test, rf_grid_predicted, average='micro'),f1_score(y_test, rf_grid_predicted, average='macro'),precision_score(y_test, rf_grid_predicted, average='micro'), precision_score(y_test, rf_grid_predicted, average='macro'),recall_score(y_test, rf_grid_predicted, average='micro'),recall_score(y_test, rf_grid_predicted, average='macro')],
         [accuracy_score(y_test, logistic_classifier_predicted_lda),f1_score(y_test, logistic_classifier_predicted_lda, average='micro'),f1_score(y_test, logistic_classifier_predicted_lda, average='macro'),precision_score(y_test, logistic_classifier_predicted_lda, average='micro'),precision_score(y_test, logistic_classifier_predicted_lda, average='macro'), recall_score(y_test, logistic_classifier_predicted_lda, average='micro'),recall_score(y_test, logistic_classifier_predicted_lda, average='macro')],
         [accuracy_score(y_test, nb_predicted_lda),f1_score(y_test, nb_predicted_lda, average='micro'),f1_score(y_test, nb_predicted_lda, average='macro'),precision_score(y_test, nb_predicted_lda, average='micro'),precision_score(y_test, nb_predicted_lda, average='macro'), recall_score(y_test, nb_predicted_lda, average='micro'),recall_score(y_test, nb_predicted_lda, average='macro')],
         [accuracy_score(y_test, rf_predicted_lda),f1_score(y_test, rf_predicted_lda, average='micro'),f1_score(y_test, rf_predicted_lda, average='macro'),precision_score(y_test, rf_predicted_lda, average='micro'),precision_score(y_test, rf_predicted_lda, average='macro'), recall_score(y_test, rf_predicted_lda, average='micro'),recall_score(y_test, rf_predicted_lda, average='macro')],
         [accuracy_score(y_test, logistic_classifier_grid_predicted_lda),f1_score(y_test, logistic_classifier_grid_predicted_lda, average='micro'),f1_score(y_test, logistic_classifier_grid_predicted_lda, average='macro'),precision_score(y_test, logistic_classifier_grid_predicted_lda, average='micro'),precision_score(y_test, logistic_classifier_grid_predicted_lda, average='macro'), recall_score(y_test, logistic_classifier_grid_predicted_lda, average='micro'),recall_score(y_test, logistic_classifier_grid_predicted_lda, average='macro')],
         [accuracy_score(y_test, nb_grid_predicted_lda),f1_score(y_test, nb_grid_predicted_lda, average='micro'),f1_score(y_test, nb_grid_predicted_lda, average='macro'),precision_score(y_test, nb_grid_predicted_lda, average='micro'),precision_score(y_test, nb_grid_predicted_lda, average='macro'), recall_score(y_test, nb_grid_predicted_lda, average='micro'),recall_score(y_test, nb_grid_predicted_lda, average='macro')],
         [accuracy_score(y_test, rf_grid_predicted_lda),f1_score(y_test, rf_grid_predicted_lda, average='micro'),f1_score(y_test, rf_grid_predicted_lda, average='macro'),precision_score(y_test, rf_grid_predicted_lda, average='micro'),precision_score(y_test, rf_grid_predicted_lda, average='macro'), recall_score(y_test, rf_grid_predicted_lda, average='micro'),recall_score(y_test, rf_grid_predicted_lda, average='macro')]]


# In[145]:


df_model_Full2_compare = pd.DataFrame(values,rows,columns)

df_model_Full2_compare


# In[146]:


import plotly.graph_objects as go

fig = go.Figure(data=[
    go.Bar(name='Micro', x=df_model_Full2_compare.index, y=df_model_Full2_compare['F1 Score micro']),
    go.Bar(name='Macro', x=df_model_Full2_compare.index, y=df_model_Full2_compare['F1 Score macro'])
])
# Change the bar mode
fig.update_layout(barmode='group',title_text='F1 Score')
fig.show()


# # 28

# In[147]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=df_model_Full2_compare.index,
    y=df_model_Full2_compare['Precision micro'],
    name='micro',
    marker_color='indianred'
))
fig.add_trace(go.Bar(
    x=df_model_Full2_compare.index,
    y=df_model_Full2_compare['Precision macro'],
    name='macro',
    marker_color='lightsalmon'
))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig.update_layout(barmode='group', xaxis_tickangle=-45,title_text='Precision Score')
fig.show()


# # 29

# In[148]:


fig = go.Figure(data=[
    go.Bar(name='Micro', x=df_model_Full2_compare.index, y=df_model_Full2_compare['Recall micro']),
    go.Bar(name='Macro', x=df_model_Full2_compare.index, y=df_model_Full2_compare['Recall macro'])
])
# Change the bar mode
fig.update_layout(barmode='group',title_text='Recall Score')
fig.show()


# # 30  เปรียบเทียบ PCA กับ LDA

# # ก่อนทำ Hyperparameter tuning

# In[157]:


rows = ['ACC LR','ACC NB','ACC RF',
        'F1 Score LR','F1 Score NB','F1 Score RF',
        'Precision LR','Precision NB','Precision RF',
        'Recall LR','Recall NB','Recall RF']
columns = ['PCA','LDA']

values = [[0.981481,0.981481],
          [0.981481,0.981481],
          [0.944444,0.944444],
          [0.981481,0.981481],
          [0.981481,0.981481],
          [0.944444,0.944444],
          [0.981481,0.981481],
          [0.981481,0.981481],
          [0.981481,0.981481],
          [0.981481,0.981481],
          [0.981481,0.981481],
          [0.981481,0.981481]]


# In[158]:


df_model_PCA_LDA_compare = pd.DataFrame(values,rows,columns)

df_model_PCA_LDA_compare


# In[160]:


fig = go.Figure(data=[
    go.Bar(name='PCA', x=df_model_PCA_LDA_compare.index, y=df_model_PCA_LDA_compare['PCA']),
    go.Bar(name='LDA', x=df_model_PCA_LDA_compare.index, y=df_model_PCA_LDA_compare['LDA'])
])
# Change the bar mode
fig.update_layout(barmode='group',title_text='PCA vs LDA Before hyperparameter tuning')
fig.show()


# # 31

# # หลังทำ Hyperparameter tuning

# In[161]:


rows = ['ACC LR HP','ACC NB HP','ACC RF HP',
        'F1 Score LR HP','F1 Score NB HP','F1 Score RF HP',
        'Precision LR HP','Precision NB HP','Precision RF HP',
        'Recall LR HP','Recall NB HP','Recall RF HP']
columns = ['PCA','LDA']

values = [[0.944444,0.981481],
          [0.981481,0.981481],
          [0.944444,0.944444],
          [0.944444,0.981481],
          [0.981481,0.981481],
          [0.944444,0.944444],
          [0.944444,0.981481],
          [0.981481,0.981481],
          [0.944444,0.944444],
          [0.944444,0.981481],
          [0.981481,0.981481],
          [0.944444,0.944444]]


# In[162]:


df_model_PCA_LDA_Grid_compare = pd.DataFrame(values,rows,columns)

df_model_PCA_LDA_Grid_compare


# In[163]:


fig = go.Figure(data=[
    go.Bar(name='PCA', x=df_model_PCA_LDA_Grid_compare.index, y=df_model_PCA_LDA_Grid_compare['PCA']),
    go.Bar(name='LDA', x=df_model_PCA_LDA_Grid_compare.index, y=df_model_PCA_LDA_Grid_compare['LDA'])
])
# Change the bar mode
fig.update_layout(barmode='group',title_text='PCA vs LDA After hyperparameter tuning')
fig.show()


# In[ ]:




