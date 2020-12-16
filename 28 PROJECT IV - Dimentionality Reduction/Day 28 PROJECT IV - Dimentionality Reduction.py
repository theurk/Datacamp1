#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # 1

# In[3]:


df = pd.read_csv('Wine_completed.csv')
df


# # 2

# In[5]:


df.head(10)


# In[6]:


df.tail(10)


# In[7]:


df.sample(10)


# # 3

# In[8]:


df.info()


# In[9]:


df.describe()


# # 4

# In[10]:


sns.pairplot(df)


# # 5

# In[11]:


df.corr()


# # 6

# In[12]:


fig = plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linecolor='white', linewidth=2)


# # 7

# In[13]:


plt.title('Best correlation')
plt.xlabel('Flavanoids')
plt.ylabel('OD280/OD315 of diluted wines')
plt.scatter(df['Flavanoids'], df['OD280/OD315 of diluted wines'])


# # 8

# In[14]:


plt.title('Worst correlation')
plt.xlabel('Malic acid')
plt.ylabel('Hue')
plt.scatter(df['Malic acid'], df['Hue'])


# # 9

# In[15]:


plt.title('Nearest zero correlation')
plt.xlabel('Ash')
plt.ylabel('OD280/OD315 of diluted wines')
plt.scatter(df['Ash'], df['OD280/OD315 of diluted wines'])


# # 10

# In[23]:


fig, [[ax1, ax2, ax3],[ ax4, ax5, ax6],[ax7, ax8, ax9]] = plt.subplots(3, 3, figsize=[12,10])
      
sns.boxplot(df['Alcohol'], orient='v', ax=ax1)
sns.boxplot(df['Malic acid'], orient='v', ax=ax2)
sns.boxplot(df['Ash'], orient='v', ax=ax3)
sns.boxplot(df['Alcalinity of ash'], orient='v', ax=ax4)
sns.boxplot(df['Magnesium'], orient='v', ax=ax5)
sns.boxplot(df['Total penols'], orient='v', ax=ax6)
sns.boxplot(df['Flavanoids'], orient='v', ax=ax7)
sns.boxplot(df['Nonflavanoids penols'], orient='v', ax=ax8)
sns.boxplot(df['Proanthocyanins'], orient='v', ax=ax9)

fig.tight_layout()


# # 11

# In[24]:


fig, [[ax1, ax2],[ ax3, ax4]] = plt.subplots(2, 2, figsize=[12,10])
      
sns.boxplot(df['Color intensity'], orient='v', ax=ax1)
sns.boxplot(df['Hue'], orient='v', ax=ax2)
sns.boxplot(df['OD280/OD315 of diluted wines'], orient='v', ax=ax3)
sns.boxplot(df['Proline'], orient='v', ax=ax4)

fig.tight_layout()


# # 12 + 13  ไม่จำเป็นต้องจัดการ Outlier  และไม่มีข้อมูลที่หายไป

# # 14

# In[27]:


#มีผลลัพธ์อยู่ใน df แล้ว  -->  df['Class']

df


# In[30]:


#ย้ายแต่ย้ายไปอยู่ column ที่ 14

df = df[[c for c in df if c not in ['Class']] + ['Class']]

df


# # 15

# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


X = df.drop(['Class'], axis = 1)
X


# In[33]:


y = df['Class']
y


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=100)


# In[35]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # 16

# In[36]:


from sklearn.preprocessing import StandardScaler


# In[37]:


sc = StandardScaler()


# In[38]:


X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# # 17  Baseline Support Vector Machine

# In[40]:


from sklearn.svm import SVC


# In[41]:


svc = SVC()


# In[42]:


svc.fit(X_train, y_train)


# In[44]:


svm_predicted = svc.predict(X_test)
svm_predicted


# # 18

# In[46]:


fig = plt.figure(figsize=(8,6))
sns.countplot(svm_predicted)


# # 19

# In[47]:


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


# In[48]:


confusion_matrix(y_test,svm_predicted)


# # 20

# In[50]:


print('Accuracy = ', accuracy_score(y_test,svm_predicted))
print('F1-score = ', f1_score(y_test,svm_predicted, average='micro'))
print('Precision = ', precision_score(y_test,svm_predicted, average='micro'))
print('Recall = ', recall_score(y_test,svm_predicted, average='micro'))


# In[51]:


print('Accuracy = ', accuracy_score(y_test,svm_predicted))
print('F1-score = ', f1_score(y_test,svm_predicted, average='macro'))
print('Precision = ', precision_score(y_test,svm_predicted, average='macro'))
print('Recall = ', recall_score(y_test,svm_predicted, average='macro'))


# # 21  Hyperparameter tuning

# In[52]:


from sklearn.model_selection import GridSearchCV


# In[53]:


param_combination = {'C':[0.01,0.1,1,10,100,1000,10000], 'gamma':[0.00001,0.0001,0.001,0.01,0.1,1,10]}


# In[54]:


grid_search = GridSearchCV(SVC(), param_combination, verbose=3)


# In[55]:


grid_search.fit(X_train, y_train)


# In[56]:


grid_search.best_params_


# In[57]:


grid_predicted = grid_search.predict(X_test)
grid_predicted


# # 22

# In[58]:


confusion_matrix(y_test,grid_predicted)


# # 23

# In[71]:


print('SVM Baseline result(micro) + Grid')

print('Accuracy = ', accuracy_score(y_test,grid_predicted))
print('F1-score = ', f1_score(y_test,grid_predicted, average='micro'))
print('Precision = ', precision_score(y_test,grid_predicted, average='micro'))
print('Recall = ', recall_score(y_test,grid_predicted, average='micro'))


# In[72]:


print('SVM Baseline result(macro) + Grid')

print('Accuracy = ', accuracy_score(y_test,grid_predicted))
print('F1-score = ', f1_score(y_test,grid_predicted, average='macro'))
print('Precision = ', precision_score(y_test,grid_predicted, average='macro'))
print('Recall = ', recall_score(y_test,grid_predicted, average='macro'))


# # 24 Random Forest Baseline

# In[61]:


from sklearn.ensemble import RandomForestClassifier


# In[62]:


rf = RandomForestClassifier()
rf.fit(X_train, y_train)


# In[63]:


rf_predicted = rf.predict(X_test)


# # 25

# In[64]:


fig = plt.figure(figsize=(8,6))
sns.countplot(rf_predicted)


# # 26

# In[65]:


confusion_matrix(y_test,rf_predicted)


# # 27

# In[66]:


print('Random Forest Baseline result(micro)')

print('Accuracy = ', accuracy_score(y_test,rf_predicted))
print('F1-score = ', f1_score(y_test,rf_predicted, average='micro'))
print('Precision = ', precision_score(y_test,rf_predicted, average='micro'))
print('Recall = ', recall_score(y_test,rf_predicted, average='micro'))


# In[67]:


print('Random Forest Baseline result(macro)')

print('Accuracy = ', accuracy_score(y_test,rf_predicted))
print('F1-score = ', f1_score(y_test,rf_predicted, average='macro'))
print('Precision = ', precision_score(y_test,rf_predicted, average='macro'))
print('Recall = ', recall_score(y_test,rf_predicted, average='macro'))


# # 28 RF + Hyperparameter tuning

# In[68]:


param_grid = {'max_depth':[4,8,16,None],
              'max_features':[4,8],
              'n_estimators':[50,100,200,500],
              'min_samples_split':[3,5,6,7]}


# In[69]:


grid_search =  GridSearchCV(RandomForestClassifier(), param_grid, verbose=3)


# In[70]:


grid_search.fit(X_train, y_train)


# In[73]:


grid_search.best_params_


# In[74]:


grid_search.best_estimator_


# In[76]:


rf_grid_predicted = grid_search.predict(X_test)


# # 29

# In[77]:


confusion_matrix(y_test,rf_grid_predicted)


# In[ ]:


#30


# In[78]:


print('Random Forest Baseline result(micro) + Grid')

print('Accuracy = ', accuracy_score(y_test,rf_grid_predicted))
print('F1-score = ', f1_score(y_test,rf_grid_predicted, average='micro'))
print('Precision = ', precision_score(y_test,rf_grid_predicted, average='micro'))
print('Recall = ', recall_score(y_test,rf_grid_predicted, average='micro'))


# In[79]:


print('Random Forest Baseline result(macro) + Grid')

print('Accuracy = ', accuracy_score(y_test,rf_grid_predicted))
print('F1-score = ', f1_score(y_test,rf_grid_predicted, average='macro'))
print('Precision = ', precision_score(y_test,rf_grid_predicted, average='macro'))
print('Recall = ', recall_score(y_test,rf_grid_predicted, average='macro'))


# # 31

# In[80]:


fig, [[ax1, ax2, ax3],[ ax4, ax5, ax6],[ax7, ax8, ax9]] = plt.subplots(3, 3, figsize=[12,10])
      
sns.distplot(df['Alcohol'], ax=ax1)
sns.distplot(df['Malic acid'], ax=ax2)
sns.distplot(df['Ash'], ax=ax3)
sns.distplot(df['Alcalinity of ash'], ax=ax4)
sns.distplot(df['Magnesium'], ax=ax5)
sns.distplot(df['Total penols'], ax=ax6)
sns.distplot(df['Flavanoids'], ax=ax7)
sns.distplot(df['Nonflavanoids penols'], ax=ax8)
sns.distplot(df['Proanthocyanins'], ax=ax9)

fig.tight_layout()


# # 32

# In[81]:


fig, [[ax1, ax2],[ ax3, ax4]] = plt.subplots(2, 2, figsize=[12,10])
      
sns.distplot(df['Color intensity'], ax=ax1)
sns.distplot(df['Hue'], ax=ax2)
sns.distplot(df['OD280/OD315 of diluted wines'], ax=ax3)
sns.distplot(df['Proline'], ax=ax4)

fig.tight_layout()


# # 33

# In[82]:


# Alcalinity of ash ใกล้เคียง Normal distribution


# # 34   LDA

# In[83]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In[84]:


lda = LDA(n_components=2)


# # 35

# In[85]:


X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)


# In[86]:


X_train_lda


# # 36

# In[87]:


fig = plt.figure(figsize=(12,8))
plt.scatter(X_train_lda[:,0],X_train_lda[:,1],c=y_train, cmap='coolwarm')
plt.xlabel('LDA 1')
plt.ylabel('LDA 2')
plt.grid()


# # 37  Support Vector Machine + LDA

# In[88]:


svc.fit(X_train_lda, y_train)


# In[89]:


svm_lda_predicted = svc.predict(X_test_lda)
svm_lda_predicted


# # 38

# In[90]:


fig = plt.figure(figsize=(8,6))
sns.countplot(svm_lda_predicted)


# # 39

# In[91]:


confusion_matrix(y_test,svm_lda_predicted)


# # 40

# In[92]:


print('SVM result(micro) + LDA')

print('Accuracy = ', accuracy_score(y_test,svm_lda_predicted))
print('F1-score = ', f1_score(y_test,svm_lda_predicted, average='micro'))
print('Precision = ', precision_score(y_test,svm_lda_predicted, average='micro'))
print('Recall = ', recall_score(y_test,svm_lda_predicted, average='micro'))


# In[93]:


print('SVM result(micro) + LDA')

print('Accuracy = ', accuracy_score(y_test,svm_lda_predicted))
print('F1-score = ', f1_score(y_test,svm_lda_predicted, average='macro'))
print('Precision = ', precision_score(y_test,svm_lda_predicted, average='macro'))
print('Recall = ', recall_score(y_test,svm_lda_predicted, average='macro'))


# # 41  Support Vector Machine + LDA + HP

# In[94]:


param_combination = {'C':[0.01,0.1,1,10,100,1000,10000], 'gamma':[0.00001,0.0001,0.001,0.01,0.1,1,10]}


# In[95]:


grid_search = GridSearchCV(SVC(), param_combination, verbose=3)


# In[96]:


grid_search.fit(X_train_lda, y_train)


# In[97]:


grid_search.best_params_


# In[98]:


grid_lda_predicted = grid_search.predict(X_test_lda)
grid_lda_predicted


# # 42

# In[99]:


confusion_matrix(y_test,grid_lda_predicted)


# # 43

# In[100]:


print('SVM result(micro) + LDA + Grid')

print('Accuracy = ', accuracy_score(y_test,grid_lda_predicted))
print('F1-score = ', f1_score(y_test,grid_lda_predicted, average='micro'))
print('Precision = ', precision_score(y_test,grid_lda_predicted, average='micro'))
print('Recall = ', recall_score(y_test,grid_lda_predicted, average='micro'))


# In[101]:


print('SVM result(micro) + LDA + Grid')

print('Accuracy = ', accuracy_score(y_test,grid_lda_predicted))
print('F1-score = ', f1_score(y_test,grid_lda_predicted, average='macro'))
print('Precision = ', precision_score(y_test,grid_lda_predicted, average='macro'))
print('Recall = ', recall_score(y_test,grid_lda_predicted, average='macro'))


# # 44 RF + LDA

# In[102]:


rf.fit(X_train_lda, y_train)


# In[103]:


rf_lda_predicted = rf.predict(X_test_lda)


# # 45

# In[104]:


fig = plt.figure(figsize=(8,6))
sns.countplot(rf_lda_predicted)


# # 46

# In[105]:


confusion_matrix(y_test,rf_lda_predicted)


# # 47

# In[106]:


print('RF result(micro) + LDA')

print('Accuracy = ', accuracy_score(y_test,rf_lda_predicted))
print('F1-score = ', f1_score(y_test,rf_lda_predicted, average='micro'))
print('Precision = ', precision_score(y_test,rf_lda_predicted, average='micro'))
print('Recall = ', recall_score(y_test,rf_lda_predicted, average='micro'))


# In[107]:


print('RF result(micro) + LDA')

print('Accuracy = ', accuracy_score(y_test,rf_lda_predicted))
print('F1-score = ', f1_score(y_test,rf_lda_predicted, average='macro'))
print('Precision = ', precision_score(y_test,rf_lda_predicted, average='macro'))
print('Recall = ', recall_score(y_test,rf_lda_predicted, average='macro'))


# # 48  RF + LDA + HP

# In[111]:


param_grid = {'max_depth':[4,8,16,None],
              'n_estimators':[50,100,200,500],
              'min_samples_split':[3,5,6,7]}


# In[112]:


grid_search =  GridSearchCV(RandomForestClassifier(), param_grid, verbose=3)


# In[113]:


grid_search.fit(X_train_lda, y_train)


# In[114]:


grid_search.best_params_


# In[115]:


grid_search.best_estimator_


# In[116]:


rf_lda_grid_predicted = grid_search.predict(X_test_lda)


# # 49

# In[117]:


confusion_matrix(y_test,rf_lda_grid_predicted)


# # 50

# In[118]:


print('RF result(micro) + LDA + Grid')

print('Accuracy = ', accuracy_score(y_test,rf_lda_grid_predicted))
print('F1-score = ', f1_score(y_test,rf_lda_grid_predicted, average='micro'))
print('Precision = ', precision_score(y_test,rf_lda_grid_predicted, average='micro'))
print('Recall = ', recall_score(y_test,rf_lda_grid_predicted, average='micro'))


# In[119]:


print('RF result(micro) + LDA + Grid')

print('Accuracy = ', accuracy_score(y_test,rf_lda_grid_predicted))
print('F1-score = ', f1_score(y_test,rf_lda_grid_predicted, average='macro'))
print('Precision = ', precision_score(y_test,rf_lda_grid_predicted, average='macro'))
print('Recall = ', recall_score(y_test,rf_lda_grid_predicted, average='macro'))


# # 51  PCA

# In[120]:


from sklearn.decomposition import PCA


# In[121]:


pca = PCA(n_components=2)


# # 52

# In[122]:


X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


# In[123]:


print(X_train_pca.shape)
print(X_test_pca.shape)


# # 52.1

# In[124]:


pca.components_


# In[126]:


X.columns


# In[127]:


df_pca = pd.DataFrame(pca.components_, columns=X.columns)
df_pca


# # 52.2

# In[130]:


fig = plt.figure(figsize=(12,8))
sns.heatmap(df_pca)


# # 53

# In[131]:


fig = plt.figure(figsize=(12,8))
plt.scatter(X_train_pca[:,0],X_train_pca[:,1],c=y_train, cmap='coolwarm')
plt.xlabel('First principle component')
plt.ylabel('Second principle component')
plt.grid()


# # 54  Support Vector Machine + PCA

# In[132]:


svc.fit(X_train_pca, y_train)


# In[133]:


svm_pca_predicted = svc.predict(X_test_pca)
svm_pca_predicted


# #    55

# In[134]:


fig = plt.figure(figsize=(8,6))
sns.countplot(svm_pca_predicted)


# # 56

# In[135]:


confusion_matrix(y_test,svm_pca_predicted)


# # 57

# In[136]:


print('SVM result(micro) + PCA')

print('Accuracy = ', accuracy_score(y_test,svm_pca_predicted))
print('F1-score = ', f1_score(y_test,svm_pca_predicted, average='micro'))
print('Precision = ', precision_score(y_test,svm_pca_predicted, average='micro'))
print('Recall = ', recall_score(y_test,svm_pca_predicted, average='micro'))


# In[138]:


print('SVM result(micro) + PCA')

print('Accuracy = ', accuracy_score(y_test,svm_pca_predicted))
print('F1-score = ', f1_score(y_test,svm_pca_predicted, average='macro'))
print('Precision = ', precision_score(y_test,svm_pca_predicted, average='macro'))
print('Recall = ', recall_score(y_test,svm_pca_predicted, average='macro'))


# # 58 Support Vector Machine + PCA +HP

# In[139]:


param_combination = {'C':[0.01,0.1,1,10,100,1000,10000], 'gamma':[0.00001,0.0001,0.001,0.01,0.1,1,10]}


# In[140]:


grid_search = GridSearchCV(SVC(), param_combination, verbose=3)


# In[141]:


grid_search.fit(X_train_pca, y_train)


# In[142]:


grid_search.best_params_


# In[143]:


grid_search.best_estimator_


# In[144]:


grid_pca_predicted = grid_search.predict(X_test_pca)
grid_pca_predicted


# # 59

# In[146]:


confusion_matrix(y_test,grid_pca_predicted)


# # 60

# In[147]:


print('SVM result(micro) + PCA + Grid')

print('Accuracy = ', accuracy_score(y_test,grid_pca_predicted))
print('F1-score = ', f1_score(y_test,grid_pca_predicted, average='micro'))
print('Precision = ', precision_score(y_test,grid_pca_predicted, average='micro'))
print('Recall = ', recall_score(y_test,grid_pca_predicted, average='micro'))


# In[150]:


print('SVM result(micro) + PCA + Grid')

print('Accuracy = ', accuracy_score(y_test,grid_pca_predicted))
print('F1-score = ', f1_score(y_test,grid_pca_predicted, average='macro'))
print('Precision = ', precision_score(y_test,grid_pca_predicted, average='macro'))
print('Recall = ', recall_score(y_test, grid_pca_predicted, average='macro'))


# # 61 RF + PCA

# In[155]:


rf.fit(X_train_pca, y_train)


# In[156]:


rf_pca_predicted = rf.predict(X_test_pca)


# # 62

# In[157]:


fig = plt.figure(figsize=(8,6))
sns.countplot(rf_pca_predicted )


# # 63

# In[158]:


confusion_matrix(y_test,rf_pca_predicted)


# # 64

# In[159]:


print('RF result(micro) + PCA')

print('Accuracy = ', accuracy_score(y_test,rf_pca_predicted ))
print('F1-score = ', f1_score(y_test,rf_pca_predicted , average='micro'))
print('Precision = ', precision_score(y_test,rf_pca_predicted , average='micro'))
print('Recall = ', recall_score(y_test,rf_pca_predicted , average='micro'))


# In[160]:


print('RF result(micro) + PCA')

print('Accuracy = ', accuracy_score(y_test,rf_pca_predicted ))
print('F1-score = ', f1_score(y_test,rf_pca_predicted , average='macro'))
print('Precision = ', precision_score(y_test,rf_pca_predicted , average='macro'))
print('Recall = ', recall_score(y_test,rf_pca_predicted , average='macro'))


# # 65 RF + PCA + HP

# In[161]:


param_grid = {'max_depth':[4,8,16,None],
              'n_estimators':[50,100,200,500],
              'min_samples_split':[3,5,6,7]}


# In[162]:


grid_search =  GridSearchCV(RandomForestClassifier(), param_grid, verbose=3)


# In[163]:


grid_search.fit(X_train_pca, y_train)


# In[164]:


grid_search.best_params_


# In[165]:


grid_search.best_estimator_


# In[166]:


rf_pca_grid_predicted = grid_search.predict(X_test_pca)


# # 66

# In[168]:


confusion_matrix(y_test,rf_pca_grid_predicted)


# # 67

# In[170]:


print('RF result(micro) + PCA + Grid')

print('Accuracy = ', accuracy_score(y_test,rf_pca_grid_predicted))
print('F1-score = ', f1_score(y_test,rf_pca_grid_predicted, average='micro'))
print('Precision = ', precision_score(y_test,rf_pca_grid_predicted, average='micro'))
print('Recall = ', recall_score(y_test,rf_pca_grid_predicted, average='micro'))


# In[171]:


print('RF result(micro) + PCA + Grid')

print('Accuracy = ', accuracy_score(y_test,rf_pca_grid_predicted))
print('F1-score = ', f1_score(y_test,rf_pca_grid_predicted, average='macro'))
print('Precision = ', precision_score(y_test,rf_pca_grid_predicted, average='macro'))
print('Recall = ', recall_score(y_test, rf_pca_grid_predicted, average='macro'))


# # 68 Before HT and Macro

# In[182]:


columns = ['method','score','score_type']
values = [['SVM',0.9861,'Accuracy'],
          ['SVM',0.9869,'F1 Score'],
          ['SVM',0.9876,'Precision'],
          ['SVM',0.9866,'Recall'],
          ['RF',0.9722,'Accuracy'],
          ['RF',0.9723,'F1 Score'],
          ['RF',0.9720,'Precision'],
          ['RF',0.9743,'Recall'],
          ['SVM+LDA',0.9583,'Accuracy'],
          ['SVM+LDA',0.9579,'F1 Score'],
          ['SVM+LDA',0.9581,'Precision'],
          ['SVM+LDA',0.9615,'Recall'],
          ['RF+LDA',0.9722,'Accuracy'],
          ['RF+LDA',0.9723,'F1 Score'],
          ['RF+LDA',0.9720,'Precision'],
          ['RF+LDA',0.9743,'Recall'],
          ['SVM+PCA',0.9583,'Accuracy'],
          ['SVM+PCA',0.9573,'F1 Score'],
          ['SVM+PCA',0.9583,'Precision'],
          ['SVM+PCA',0.9615,'Recall'],
          ['RF+PCA',0.8888,'Accuracy'],
          ['RF+PCA',0.8893,'F1 Score'],
          ['RF+PCA',0.8907,'Precision'],
          ['RF+PCA',0.8903,'Recall']]


# In[183]:


df_results_before_HT = pd.DataFrame(values, columns=columns)

df_results_before_HT


# In[184]:


fig = plt.figure(figsize=(8,6))
sns.barplot(x='method', y='score', hue='score_type', data=df_results_before_HT)


# # 69 After HT and Micro

# In[185]:


columns = ['method','score','score_type']
values = [['SVM',0.9583,'Accuracy'],
          ['SVM',0.9583,'F1 Score'],
          ['SVM',0.9583,'Precision'],
          ['SVM',0.9583,'Recall'],
          ['RF',0.9861,'Accuracy'],
          ['RF',0.9861,'F1 Score'],
          ['RF',0.9861,'Precision'],
          ['RF',0.9861,'Recall'],
          ['SVM+LDA',0.9583,'Accuracy'],
          ['SVM+LDA',0.9583,'F1 Score'],
          ['SVM+LDA',0.9583,'Precision'],
          ['SVM+LDA',0.9583,'Recall'],
          ['RF+LDA',0.9583,'Accuracy'],
          ['RF+LDA',0.9583,'F1 Score'],
          ['RF+LDA',0.9583,'Precision'],
          ['RF+LDA',0.9583,'Recall'],
          ['SVM+PCA',0.9583,'Accuracy'],
          ['SVM+PCA',0.9583,'F1 Score'],
          ['SVM+PCA',0.9583,'Precision'],
          ['SVM+PCA',0.9583,'Recall'],
          ['RF+PCA',0.9583,'Accuracy'],
          ['RF+PCA',0.9583,'F1 Score'],
          ['RF+PCA',0.9583,'Precision'],
          ['RF+PCA',0.9583,'Recall']]


# In[186]:


df_results_After_HT = pd.DataFrame(values, columns=columns)

df_results_After_HT


# In[188]:


fig = plt.figure(figsize=(8,6))
sns.barplot(x='method', y='score', hue='score_type', data=df_results_After_HT)


# # 70 

# In[189]:


import plotly.express as px


# In[190]:


columns = ['method','score','score_type']
values = [['SVM+LDA',0.9583,'Accuracy'],
          ['RF+LDA',0.9583,'Accuracy'],
          ['SVM+PCA',0.9583,'Accuracy'],
          ['RF+PCA',0.9583,'Accuracy']]


df_results = pd.DataFrame(values, columns=columns)

df_results


# In[191]:


fig = px.bar(df_results, x='method', y='score', title='ACC Comparision')
fig.show()


# In[ ]:




