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

# In[8]:


df.info()


# In[9]:


df.describe()


# # 5

# In[10]:


sns.pairplot(df)


# # 6

# In[14]:


sns.distplot(df['Alcohol'])


# In[15]:


sns.distplot(df['Malic acid'])


# In[16]:


sns.distplot(df['Ash'])


# In[17]:


sns.distplot(df['Alcalinity of ash'])


# In[18]:


sns.distplot(df['Magnesium'])


# In[19]:


sns.distplot(df['Flavanoids'])


# # 7

# In[20]:


df.corr()


# In[21]:


fig = plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linecolor='white', linewidth=2)


# # 8

# In[22]:


plt.title('Best correlation')
plt.xlabel('Flavanoids')
plt.ylabel('OD280/OD315 of diluted wines')
plt.scatter(df['Flavanoids'], df['OD280/OD315 of diluted wines'])


# # 9

# In[24]:


plt.title('Worst correlation')
plt.xlabel('Malic acid')
plt.ylabel('Hue')
plt.scatter(df['Malic acid'], df['Hue'])


# # 10

# In[25]:


fig = plt.figure(figsize=(12,8))
plt.hist(df['Malic acid'],bins=10)
plt.show()


# In[26]:


fig = plt.figure(figsize=(12,8))
plt.hist(df['Flavanoids'],bins=10)
plt.show()


# # 11

# In[27]:


fig = plt.figure(figsize=(12,8))
sns.boxplot(x='Class',y='Malic acid', data=df)


# In[28]:


fig = plt.figure(figsize=(12,8))
sns.boxplot(x='Class',y='Flavanoids', data=df)


# In[29]:


fig = plt.figure(figsize=(12,8))
sns.boxplot(x='Class',y='Magnesium', data=df)


# # 12

# In[30]:


#ไม่ได้จัดการ Outlier เนื่องจากขาดความรู้ในเรื่องสารเคมีต่างๆครับ


# # 13

# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


df


# In[33]:


X = df.drop(['Class'], axis = 1)
X


# In[34]:


y = df['Class']
y


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


# In[36]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # 14

# In[38]:


from sklearn.preprocessing import StandardScaler


# In[40]:


sc = StandardScaler()


# In[43]:


X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[44]:


X_train


# # 15  

# In[45]:


# เลือก Baseline 3 ชนิด

# 1. Logistic Regression
# 2. Naive Bayes
# 3. Random Forest


# # 16   Baseline 1 Logistic Regression

# In[46]:


from sklearn.linear_model import LogisticRegression


# In[47]:


logistic_classifier = LogisticRegression()
logistic_classifier.fit(X_train, y_train)


# In[48]:


logistic_classifier_predicted = logistic_classifier.predict(X_test)


# # 17

# In[49]:


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


# In[51]:


confusion_matrix(y_test,logistic_classifier_predicted)


# In[56]:


print('Logistic Regression Result(micro)')

print('Accuracy = ', accuracy_score(y_test,logistic_classifier_predicted))
print('F1-score = ', f1_score(y_test,logistic_classifier_predicted, average='micro'))
print('Precision = ', precision_score(y_test,logistic_classifier_predicted, average='micro'))
print('Recall = ', recall_score(y_test,logistic_classifier_predicted, average='micro'))


# In[57]:


print('Logistic Regression Result(macro)')

print('Accuracy = ', accuracy_score(y_test,logistic_classifier_predicted))
print('F1-score = ', f1_score(y_test,logistic_classifier_predicted, average='macro'))
print('Precision = ', precision_score(y_test,logistic_classifier_predicted, average='macro'))
print('Recall = ', recall_score(y_test,logistic_classifier_predicted, average='macro'))


# # Baseline 2 Naive Bayes

# In[59]:


from sklearn.naive_bayes import GaussianNB


# In[60]:


nb = GaussianNB()
nb.fit(X_train, y_train)


# In[62]:


nb_predicted = nb.predict(X_test)


# In[63]:


confusion_matrix(y_test,nb_predicted)


# In[64]:


print('Naive Bayes Result(micro)')

print('Accuracy = ', accuracy_score(y_test,nb_predicted))
print('F1-score = ', f1_score(y_test,nb_predicted, average='micro'))
print('Precision = ', precision_score(y_test,nb_predicted, average='micro'))
print('Recall = ', recall_score(y_test,nb_predicted, average='micro'))


# In[65]:


print('Naive Bayes Result(macro)')

print('Accuracy = ', accuracy_score(y_test,nb_predicted))
print('F1-score = ', f1_score(y_test,nb_predicted, average='macro'))
print('Precision = ', precision_score(y_test,nb_predicted, average='macro'))
print('Recall = ', recall_score(y_test,nb_predicted, average='macro'))


# # Baseline 3 Random Forest

# In[66]:


from sklearn.ensemble import RandomForestClassifier


# In[68]:


rf = RandomForestClassifier()
rf.fit(X_train, y_train)


# In[69]:


rf_predicted = rf.predict(X_test)


# In[70]:


confusion_matrix(y_test,rf_predicted)


# In[71]:


print('Random Forest Result(micro)')

print('Accuracy = ', accuracy_score(y_test,rf_predicted))
print('F1-score = ', f1_score(y_test,rf_predicted, average='micro'))
print('Precision = ', precision_score(y_test,rf_predicted, average='micro'))
print('Recall = ', recall_score(y_test,rf_predicted, average='micro'))


# In[72]:


print('Random Forest Result(macro)')

print('Accuracy = ', accuracy_score(y_test,rf_predicted))
print('F1-score = ', f1_score(y_test,rf_predicted, average='macro'))
print('Precision = ', precision_score(y_test,rf_predicted, average='macro'))
print('Recall = ', recall_score(y_test,rf_predicted, average='macro'))


# # 18  ทำ Grid Search

# In[73]:


from sklearn.model_selection import GridSearchCV


# # Logistic Regression + GridSearch

# In[118]:


param_grid = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}


# In[119]:


grid_search =  GridSearchCV(LogisticRegression(), param_grid, verbose=3)


# In[120]:


grid_search.fit(X_train, y_train)


# In[121]:


grid_search.best_params_


# In[122]:


grid_search.best_estimator_


# In[123]:


logistic_classifier_grid_predicted = grid_search.predict(X_test)


# # 19 วัดผล หลังทำ Parameter tuning

# In[124]:


confusion_matrix(y_test,logistic_classifier_grid_predicted)


# In[125]:


print('Logistic Regression + GridSearch Result(micro)')

print('Accuracy = ', accuracy_score(y_test,logistic_classifier_grid_predicted))
print('F1-score = ', f1_score(y_test,logistic_classifier_grid_predicted, average='micro'))
print('Precision = ', precision_score(y_test,logistic_classifier_grid_predicted, average='micro'))
print('Recall = ', recall_score(y_test,logistic_classifier_grid_predicted, average='micro'))


# In[126]:


print('Logistic Regression + GridSearch Result(macro)')

print('Accuracy = ', accuracy_score(y_test,logistic_classifier_grid_predicted))
print('F1-score = ', f1_score(y_test,logistic_classifier_grid_predicted, average='macro'))
print('Precision = ', precision_score(y_test,logistic_classifier_grid_predicted, average='macro'))
print('Recall = ', recall_score(y_test,logistic_classifier_grid_predicted, average='macro'))


# # Naive Bayes + GridSearch

# In[109]:


param_grid = {'var_smoothing' : [1e-09,2e-05,1e-5,1e-7,3e-1]}


# In[110]:


grid_search =  GridSearchCV(GaussianNB(), param_grid, verbose=3)


# In[111]:


grid_search.fit(X_train, y_train)


# In[112]:


grid_search.best_params_


# In[113]:


grid_search.best_estimator_


# In[114]:


nb_grid_predicted = grid_search.predict(X_test)


# In[115]:


confusion_matrix(y_test,nb_grid_predicted)


# In[116]:


print('Naive Bayes + GridSearch Result(micro)')

print('Accuracy = ', accuracy_score(y_test,nb_grid_predicted))
print('F1-score = ', f1_score(y_test,nb_grid_predicted, average='micro'))
print('Precision = ', precision_score(y_test,nb_grid_predicted, average='micro'))
print('Recall = ', recall_score(y_test,nb_grid_predicted, average='micro'))


# In[117]:


print('Naive Bayes + GridSearch Result(macro)')

print('Accuracy = ', accuracy_score(y_test,nb_grid_predicted))
print('F1-score = ', f1_score(y_test,nb_grid_predicted, average='macro'))
print('Precision = ', precision_score(y_test,nb_grid_predicted, average='macro'))
print('Recall = ', recall_score(y_test,nb_grid_predicted, average='macro'))


# # Random Forest + GridSearch

# In[128]:


param_grid = {'max_depth':[4,8,16,None],
              'max_features':[4,8],
              'n_estimators':[50,100,200,500],
              'min_samples_split':[3,5,6,7]}


# In[129]:


grid_search =  GridSearchCV(RandomForestClassifier(), param_grid, verbose=3)


# In[130]:


grid_search.fit(X_train, y_train)


# In[131]:


grid_search.best_params_


# In[132]:


grid_search.best_estimator_


# In[133]:


rf_grid_predicted = grid_search.predict(X_test)


# In[134]:


print('Naive Bayes + GridSearch Result(micro)')

print('Accuracy = ', accuracy_score(y_test,rf_grid_predicted))
print('F1-score = ', f1_score(y_test,rf_grid_predicted, average='micro'))
print('Precision = ', precision_score(y_test,rf_grid_predicted, average='micro'))
print('Recall = ', recall_score(y_test,rf_grid_predicted, average='micro'))


# In[135]:


print('Naive Bayes + GridSearch Result(macro)')

print('Accuracy = ', accuracy_score(y_test,rf_grid_predicted))
print('F1-score = ', f1_score(y_test,rf_grid_predicted, average='macro'))
print('Precision = ', precision_score(y_test,rf_grid_predicted, average='macro'))
print('Recall = ', recall_score(y_test,rf_grid_predicted, average='macro'))


# # 20 PCA

# In[136]:


from sklearn.decomposition import PCA


# In[138]:


pca = PCA(n_components=2)


# In[139]:


X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


# In[140]:


print(X_train_pca.shape)
print(X_test_pca.shape)


# # 21

# In[141]:


pca.components_


# In[142]:


df_comp = pd.DataFrame(pca.components_, columns=X.columns)
df_comp


# # 22

# In[144]:


fig = plt.figure(figsize=(12,8))
sns.heatmap(df_comp, annot=True, cmap='coolwarm', linecolor='white', linewidth=2)


# In[145]:


pca.explained_variance_


# In[146]:


pca.n_components


# # 23

# In[147]:


X_train_pca


# In[148]:


X_train_pca[:,0]


# In[149]:


fig = plt.figure(figsize=(12,8))
plt.scatter(X_train_pca[:,0],X_train_pca[:,1],c=y_train, cmap='coolwarm')
plt.xlabel('First principle component')
plt.ylabel('Second principle component')
plt.grid()


# # 24 เทรนโมเดล + PCA

# # Logistic Regression

# In[151]:


logistic_classifier_pca = LogisticRegression()
logistic_classifier_pca.fit(X_train_pca, y_train)


# In[153]:


logistic_classifier_predicted_pca = logistic_classifier_pca.predict(X_test_pca)


# # 25 วัดผล

# In[154]:


confusion_matrix(y_test,logistic_classifier_predicted_pca)


# In[155]:


print('Logistic Regression Result(micro) + PCA')

print('Accuracy = ', accuracy_score(y_test,logistic_classifier_predicted_pca))
print('F1-score = ', f1_score(y_test,logistic_classifier_predicted_pca, average='micro'))
print('Precision = ', precision_score(y_test,logistic_classifier_predicted_pca, average='micro'))
print('Recall = ', recall_score(y_test,logistic_classifier_predicted_pca, average='micro'))


# In[157]:


print('Logistic Regression Result(macro) + PCA')

print('Accuracy = ', accuracy_score(y_test,logistic_classifier_predicted_pca))
print('F1-score = ', f1_score(y_test,logistic_classifier_predicted_pca, average='macro'))
print('Precision = ', precision_score(y_test,logistic_classifier_predicted_pca, average='macro'))
print('Recall = ', recall_score(y_test,logistic_classifier_predicted_pca, average='macro'))


# # Naive Bayes

# In[158]:


nb_pca = GaussianNB()
nb_pca.fit(X_train_pca, y_train)


# In[160]:


nb_predicted_pca = nb_pca.predict(X_test_pca)


# In[161]:


confusion_matrix(y_test,nb_predicted_pca)


# In[164]:


print('Naive Bayes Result(micro) + PCA')

print('Accuracy = ', accuracy_score(y_test,nb_predicted_pca))
print('F1-score = ', f1_score(y_test,nb_predicted_pca, average='micro'))
print('Precision = ', precision_score(y_test,nb_predicted_pca, average='micro'))
print('Recall = ', recall_score(y_test,nb_predicted_pca, average='micro'))


# In[165]:


print('Naive Bayes Result(macro) + PCA')

print('Accuracy = ', accuracy_score(y_test,nb_predicted_pca))
print('F1-score = ', f1_score(y_test,nb_predicted_pca, average='macro'))
print('Precision = ', precision_score(y_test,nb_predicted_pca, average='macro'))
print('Recall = ', recall_score(y_test,nb_predicted_pca, average='macro'))


# # Random Forest

# In[166]:


rf_pca = RandomForestClassifier()
rf_pca.fit(X_train_pca, y_train)


# In[167]:


rf_predicted_pca = rf_pca.predict(X_test_pca)


# In[168]:


confusion_matrix(y_test,rf_predicted_pca)


# In[169]:


print('Random Forest Result(micro) + PCA')

print('Accuracy = ', accuracy_score(y_test,rf_predicted_pca))
print('F1-score = ', f1_score(y_test,rf_predicted_pca, average='micro'))
print('Precision = ', precision_score(y_test,rf_predicted_pca, average='micro'))
print('Recall = ', recall_score(y_test,rf_predicted_pca, average='micro'))


# In[170]:


print('Random Forest Result(macro) + PCA')

print('Accuracy = ', accuracy_score(y_test,rf_predicted_pca))
print('F1-score = ', f1_score(y_test,rf_predicted_pca, average='macro'))
print('Precision = ', precision_score(y_test,rf_predicted_pca, average='macro'))
print('Recall = ', recall_score(y_test,rf_predicted_pca, average='macro'))


# # 26  parameter tuning กับ โมเดลใหม่(PCA)

# # Logistic Regression(PCA) + GridSearch

# In[174]:


param_grid = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}


# In[175]:


grid_search =  GridSearchCV(LogisticRegression(), param_grid, verbose=3)


# In[176]:


grid_search.fit(X_train_pca, y_train)


# In[177]:


grid_search.best_params_


# In[178]:


grid_search.best_estimator_


# In[179]:


logistic_classifier_grid_predicted_pca = grid_search.predict(X_test_pca)


# # 27 วัดผล

# In[180]:


confusion_matrix(y_test,logistic_classifier_grid_predicted_pca)


# In[181]:


print('Logistic Regression + GridSearch Result(micro) + PCA')

print('Accuracy = ', accuracy_score(y_test,logistic_classifier_grid_predicted_pca))
print('F1-score = ', f1_score(y_test,logistic_classifier_grid_predicted_pca, average='micro'))
print('Precision = ', precision_score(y_test,logistic_classifier_grid_predicted_pca, average='micro'))
print('Recall = ', recall_score(y_test,logistic_classifier_grid_predicted_pca, average='micro'))


# In[182]:


print('Logistic Regression + GridSearch Result(macro) + PCA')

print('Accuracy = ', accuracy_score(y_test,logistic_classifier_grid_predicted_pca))
print('F1-score = ', f1_score(y_test,logistic_classifier_grid_predicted_pca, average='macro'))
print('Precision = ', precision_score(y_test,logistic_classifier_grid_predicted_pca, average='macro'))
print('Recall = ', recall_score(y_test,logistic_classifier_grid_predicted_pca, average='macro'))


# # Naive Bayes(PCA) + GridSearch

# In[183]:


param_grid = {'var_smoothing' : [1e-09,2e-05,1e-5,1e-7,3e-1]}


# In[184]:


grid_search =  GridSearchCV(GaussianNB(), param_grid, verbose=3)


# In[185]:


grid_search.fit(X_train_pca, y_train)


# In[186]:


grid_search.best_params_


# In[187]:


grid_search.best_estimator_


# In[189]:


nb_grid_predicted_pca = grid_search.predict(X_test_pca)


# In[190]:


confusion_matrix(y_test,nb_grid_predicted_pca)


# In[191]:


print('Naive Bayes + GridSearch Result(micro) + PCA')

print('Accuracy = ', accuracy_score(y_test,nb_grid_predicted_pca))
print('F1-score = ', f1_score(y_test,nb_grid_predicted_pca, average='micro'))
print('Precision = ', precision_score(y_test,nb_grid_predicted_pca, average='micro'))
print('Recall = ', recall_score(y_test,nb_grid_predicted_pca, average='micro'))


# In[192]:


print('Naive Bayes + GridSearch Result(macro) + PCA')

print('Accuracy = ', accuracy_score(y_test,nb_grid_predicted_pca))
print('F1-score = ', f1_score(y_test,nb_grid_predicted_pca, average='macro'))
print('Precision = ', precision_score(y_test,nb_grid_predicted_pca, average='macro'))
print('Recall = ', recall_score(y_test,nb_grid_predicted_pca, average='macro'))


# # Random Forest(PCA) + GridSearch

# In[196]:


param_grid = {'max_depth':[4,8,16,None],
              'max_features':[2],
              'n_estimators':[50,100,200,500],
              'min_samples_split':[3,5,6,7]}


# In[197]:


grid_search =  GridSearchCV(RandomForestClassifier(), param_grid, verbose=3)


# In[198]:


grid_search.fit(X_train_pca, y_train)


# In[199]:


grid_search.best_params_


# In[200]:


grid_search.best_estimator_


# In[201]:


rf_grid_predicted_pca = grid_search.predict(X_test_pca)


# In[202]:


confusion_matrix(y_test,rf_grid_predicted_pca)


# In[203]:


print('RandomForest + GridSearch Result(micro) + PCA')

print('Accuracy = ', accuracy_score(y_test,rf_grid_predicted_pca))
print('F1-score = ', f1_score(y_test,rf_grid_predicted_pca, average='micro'))
print('Precision = ', precision_score(y_test,rf_grid_predicted_pca, average='micro'))
print('Recall = ', recall_score(y_test,rf_grid_predicted_pca, average='micro'))


# In[204]:


print('RandomForest + GridSearch Result(macro) + PCA')

print('Accuracy = ', accuracy_score(y_test,rf_grid_predicted_pca))
print('F1-score = ', f1_score(y_test,rf_grid_predicted_pca, average='macro'))
print('Precision = ', precision_score(y_test,rf_grid_predicted_pca, average='macro'))
print('Recall = ', recall_score(y_test,rf_grid_predicted_pca, average='macro'))


# # 28

# In[205]:


#Micro-averaging and macro-averaging scoring metrics is used for evaluating models trained for multi-class classification problems.

#Macro-averaging scores are arithmetic mean of individual classes’ score in relation to precision, recall and f1-score

#Micro-averaging precision scores is sum of true positive for individual classes divided by sum of predicted positives for all classes

#Micro-averaging recall scores is sum of true positive for individual classes divided by sum of actual positives for all classes

#Use micro-averaging score when there is a need to weight each instance or prediction equally.

#Use macro-averaging score when all classes need to be treated equally to evaluate the overall performance of the classifier with regard to the most frequent class labels.

#Use weighted macro-averaging score in case of class imbalances (different number of instances related to different class labels).


# # 29

# In[206]:


import plotly.express as px


# In[208]:


rows = ['LR','NB','RF','LR_HT','NB_HT','RF_HT','LR_PCA','NB_PCA','RF_PCA','LR_PCA_HT','NB_PCA_HT','RF_PCA_HT']
columns = ['ACC','F1 Score','Precision','Recall']

values = [[accuracy_score(y_test, logistic_classifier_predicted),f1_score(y_test, logistic_classifier_predicted, average='micro'),precision_score(y_test, logistic_classifier_predicted, average='micro'), recall_score(y_test, logistic_classifier_predicted, average='micro')],
         [accuracy_score(y_test, nb_predicted),f1_score(y_test, nb_predicted, average='micro'),precision_score(y_test, nb_predicted, average='micro'), recall_score(y_test, nb_predicted, average='micro')],
         [accuracy_score(y_test, rf_predicted),f1_score(y_test, rf_predicted, average='micro'),precision_score(y_test, rf_predicted, average='micro'), recall_score(y_test, rf_predicted, average='micro')],
         [accuracy_score(y_test, logistic_classifier_grid_predicted),f1_score(y_test, logistic_classifier_grid_predicted, average='micro'),precision_score(y_test, logistic_classifier_grid_predicted, average='micro'), recall_score(y_test, logistic_classifier_grid_predicted, average='micro')],
         [accuracy_score(y_test, nb_grid_predicted),f1_score(y_test, nb_grid_predicted, average='micro'),precision_score(y_test, nb_grid_predicted, average='micro'), recall_score(y_test, nb_grid_predicted, average='micro')],
         [accuracy_score(y_test, rf_grid_predicted),f1_score(y_test, rf_grid_predicted, average='micro'),precision_score(y_test, rf_grid_predicted, average='micro'), recall_score(y_test, rf_grid_predicted, average='micro')],
         [accuracy_score(y_test, logistic_classifier_predicted_pca),f1_score(y_test, logistic_classifier_predicted_pca, average='micro'),precision_score(y_test, logistic_classifier_predicted_pca, average='micro'), recall_score(y_test, logistic_classifier_predicted_pca, average='micro')],
         [accuracy_score(y_test, nb_predicted_pca),f1_score(y_test, nb_predicted_pca, average='micro'),precision_score(y_test, nb_predicted_pca, average='micro'), recall_score(y_test, nb_predicted_pca, average='micro')],
         [accuracy_score(y_test, rf_predicted_pca),f1_score(y_test, rf_predicted_pca, average='micro'),precision_score(y_test, rf_predicted_pca, average='micro'), recall_score(y_test, rf_predicted_pca, average='micro')],
         [accuracy_score(y_test, logistic_classifier_grid_predicted_pca),f1_score(y_test, logistic_classifier_grid_predicted_pca, average='micro'),precision_score(y_test, logistic_classifier_grid_predicted_pca, average='micro'), recall_score(y_test, logistic_classifier_grid_predicted_pca, average='micro')],
         [accuracy_score(y_test, nb_grid_predicted_pca),f1_score(y_test, nb_grid_predicted_pca, average='micro'),precision_score(y_test, nb_grid_predicted_pca, average='micro'), recall_score(y_test, nb_grid_predicted_pca, average='micro')],
         [accuracy_score(y_test, rf_grid_predicted_pca),f1_score(y_test, rf_grid_predicted_pca, average='micro'),precision_score(y_test, rf_grid_predicted_pca, average='micro'), recall_score(y_test, rf_grid_predicted_pca, average='micro')]]


# In[209]:


df_model_Full_compare = pd.DataFrame(values,rows,columns)

df_model_Full_compare


# In[211]:


fig = px.bar(df_model_Full_compare, y='ACC', x=df_model_Full_compare.index, title='ACC Comparison')
fig.show()


# # 30

# In[212]:


rows = ['LR','NB','RF','LR_HT','NB_HT','RF_HT','LR_PCA','NB_PCA','RF_PCA','LR_PCA_HT','NB_PCA_HT','RF_PCA_HT']
columns = ['ACC','F1 Score micro','F1 Score macro','Precision micro','Precision macro','Recall micro','Recall macro']

values = [[accuracy_score(y_test, logistic_classifier_predicted),f1_score(y_test, logistic_classifier_predicted, average='micro'),f1_score(y_test, logistic_classifier_predicted, average='macro'),precision_score(y_test, logistic_classifier_predicted, average='micro'),precision_score(y_test, logistic_classifier_predicted, average='macro'), recall_score(y_test, logistic_classifier_predicted, average='micro'),recall_score(y_test, logistic_classifier_predicted, average='macro')],
         [accuracy_score(y_test, nb_predicted),f1_score(y_test, nb_predicted, average='micro'),f1_score(y_test, nb_predicted, average='macro'),precision_score(y_test, nb_predicted, average='micro'),precision_score(y_test, nb_predicted, average='macro'), recall_score(y_test, nb_predicted, average='micro'),recall_score(y_test, nb_predicted, average='macro')],
         [accuracy_score(y_test, rf_predicted),f1_score(y_test, rf_predicted, average='micro'),f1_score(y_test, rf_predicted, average='macro'),precision_score(y_test, rf_predicted, average='micro'),precision_score(y_test, rf_predicted, average='macro'), recall_score(y_test, rf_predicted, average='micro'),recall_score(y_test, rf_predicted, average='macro')],
         [accuracy_score(y_test, logistic_classifier_grid_predicted),f1_score(y_test, logistic_classifier_grid_predicted, average='micro'),f1_score(y_test, logistic_classifier_grid_predicted, average='macro'),precision_score(y_test, logistic_classifier_grid_predicted, average='micro'),precision_score(y_test, logistic_classifier_grid_predicted, average='macro'), recall_score(y_test, logistic_classifier_grid_predicted, average='micro'),recall_score(y_test, logistic_classifier_grid_predicted, average='macro')],
         [accuracy_score(y_test, nb_grid_predicted),f1_score(y_test, nb_grid_predicted, average='micro'),f1_score(y_test, nb_grid_predicted, average='macro'),precision_score(y_test, nb_grid_predicted, average='micro'), precision_score(y_test, nb_grid_predicted, average='macro'),recall_score(y_test, nb_grid_predicted, average='micro'),recall_score(y_test, nb_grid_predicted, average='macro')],
         [accuracy_score(y_test, rf_grid_predicted),f1_score(y_test, rf_grid_predicted, average='micro'),f1_score(y_test, rf_grid_predicted, average='macro'),precision_score(y_test, rf_grid_predicted, average='micro'), precision_score(y_test, rf_grid_predicted, average='macro'),recall_score(y_test, rf_grid_predicted, average='micro'),recall_score(y_test, rf_grid_predicted, average='macro')],
         [accuracy_score(y_test, logistic_classifier_predicted_pca),f1_score(y_test, logistic_classifier_predicted_pca, average='micro'),f1_score(y_test, logistic_classifier_predicted_pca, average='macro'),precision_score(y_test, logistic_classifier_predicted_pca, average='micro'),precision_score(y_test, logistic_classifier_predicted_pca, average='macro'), recall_score(y_test, logistic_classifier_predicted_pca, average='micro'),recall_score(y_test, logistic_classifier_predicted_pca, average='macro')],
         [accuracy_score(y_test, nb_predicted_pca),f1_score(y_test, nb_predicted_pca, average='micro'),f1_score(y_test, nb_predicted_pca, average='macro'),precision_score(y_test, nb_predicted_pca, average='micro'),precision_score(y_test, nb_predicted_pca, average='macro'), recall_score(y_test, nb_predicted_pca, average='micro'),recall_score(y_test, nb_predicted_pca, average='macro')],
         [accuracy_score(y_test, rf_predicted_pca),f1_score(y_test, rf_predicted_pca, average='micro'),f1_score(y_test, rf_predicted_pca, average='macro'),precision_score(y_test, rf_predicted_pca, average='micro'),precision_score(y_test, rf_predicted_pca, average='macro'), recall_score(y_test, rf_predicted_pca, average='micro'),recall_score(y_test, rf_predicted_pca, average='macro')],
         [accuracy_score(y_test, logistic_classifier_grid_predicted_pca),f1_score(y_test, logistic_classifier_grid_predicted_pca, average='micro'),f1_score(y_test, logistic_classifier_grid_predicted_pca, average='macro'),precision_score(y_test, logistic_classifier_grid_predicted_pca, average='micro'),precision_score(y_test, logistic_classifier_grid_predicted_pca, average='macro'), recall_score(y_test, logistic_classifier_grid_predicted_pca, average='micro'),recall_score(y_test, logistic_classifier_grid_predicted_pca, average='macro')],
         [accuracy_score(y_test, nb_grid_predicted_pca),f1_score(y_test, nb_grid_predicted_pca, average='micro'),f1_score(y_test, nb_grid_predicted_pca, average='macro'),precision_score(y_test, nb_grid_predicted_pca, average='micro'),precision_score(y_test, nb_grid_predicted_pca, average='macro'), recall_score(y_test, nb_grid_predicted_pca, average='micro'),recall_score(y_test, nb_grid_predicted_pca, average='macro')],
         [accuracy_score(y_test, rf_grid_predicted_pca),f1_score(y_test, rf_grid_predicted_pca, average='micro'),f1_score(y_test, rf_grid_predicted_pca, average='macro'),precision_score(y_test, rf_grid_predicted_pca, average='micro'),precision_score(y_test, rf_grid_predicted_pca, average='macro'), recall_score(y_test, rf_grid_predicted_pca, average='micro'),recall_score(y_test, rf_grid_predicted_pca, average='macro')]]


# In[213]:


df_model_Full2_compare = pd.DataFrame(values,rows,columns)

df_model_Full2_compare


# In[221]:


import plotly.graph_objects as go

fig = go.Figure(data=[
    go.Bar(name='Micro', x=df_model_Full2_compare.index, y=df_model_Full2_compare['F1 Score micro']),
    go.Bar(name='Macro', x=df_model_Full2_compare.index, y=df_model_Full2_compare['F1 Score macro'])
])
# Change the bar mode
fig.update_layout(barmode='group',title_text='F1 Score')
fig.show()


# # 31

# In[223]:


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


# # 32

# In[224]:


fig = go.Figure(data=[
    go.Bar(name='Micro', x=df_model_Full2_compare.index, y=df_model_Full2_compare['Recall micro']),
    go.Bar(name='Macro', x=df_model_Full2_compare.index, y=df_model_Full2_compare['Recall macro'])
])
# Change the bar mode
fig.update_layout(barmode='group',title_text='Recall Score')
fig.show()


# In[ ]:




