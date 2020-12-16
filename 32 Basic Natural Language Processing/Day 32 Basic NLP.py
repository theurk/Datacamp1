#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # 1

# In[7]:


df = pd.read_csv('yelp.csv')
df


# # 2

# In[8]:


df.head(10)


# In[9]:


df.tail(10)


# In[10]:


df.sample(10)


# # 3

# In[11]:


df.info()


# In[12]:


df.describe()


# # 4

# In[13]:


sns.pairplot(df)


# # 5

# In[16]:


df['year'] = pd.DatetimeIndex(df['date']).year


# In[17]:


df


# In[18]:


df['year'].value_counts()


# # 6

# In[21]:


df['month'] = pd.DatetimeIndex(df['date']).month


# In[24]:


df['month'].value_counts()


# # 7

# In[56]:


df2 = df.groupby(["month", "year"]).size()

df2


# In[57]:


df.groupby(["month", "year"]).size().max()


# In[58]:


df.groupby(["month", "year"]).size() == 304


# In[61]:


df2[df.groupby(["month", "year"]).size() == 304]


# # 8

# In[63]:


sns.countplot(df['stars'])


# # 9

# In[64]:


df.corr()


# # 10    ทำแบบค่า Sum

# In[65]:


import plotly.express as px


# In[67]:


fig = px.pie(df, values='cool', names='stars', title='Sum of Cool')

fig.show()


# # 11  ทำแบบค่า Mean

# In[76]:


df['stars'].value_counts()


# In[77]:


df.groupby('stars').mean()


# In[78]:


df_temp = df.groupby('stars').mean().reset_index()
df_temp


# In[79]:


fig = px.pie(df_temp, values='useful', names='stars', title='Mean of Useful')

fig.show()


# # 12

# In[82]:


fig = px.pie(df_temp, values='funny', names='stars', title='Mean of Funny')

fig.show()


# # 13  หาความยาวของ text แต่ละแถว แล้วเพิ่มเป็นคอลัมใหม่

# In[83]:


df['text'].apply(len)


# In[84]:


df['Length'] = df['text'].apply(len)


# In[85]:


df


# # 14   plot คนละ subplot

# In[86]:


df.hist(column='Length', by='stars', bins=100, figsize=(14,8))


# # 15  % sparsity

# In[90]:


#ยังหาค่าไม่ได้ เนื่องจากยังไม่ได้ทำ Bag Of Words


# # 16  ลบ punctuation

# In[92]:


import string


# In[94]:


string.punctuation


# In[95]:


df


# In[97]:


no_punc = [x for x in df['text'] if x not in string.punctuation]

no_punc


# In[104]:


no_punc = ''.join(no_punc)

no_punc


# # 17 ลบ stopwords

# In[105]:


from nltk.corpus import stopwords


# In[106]:


import nltk
nltk.download('stopwords')


# In[107]:


len(stopwords.words('english'))


# In[108]:


stopwords.words('english')


# In[109]:


no_punc.split()


# In[111]:


[word for word in no_punc.split() if word.lower() not in stopwords.words('english')]


# # 18  เขียนฟังก์ชั่นรวมมิตรเลย

# In[142]:


def text_filtering(text_text):
    #ลบ punctuation
    no_punc = [x for x in text_text if x not in string.punctuation]
    
    #นำมารวมกันใหม่ ใช้ join
    no_punc = ''.join(no_punc)
    
    #ลบ stopwords
    text = [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]
    
    return text


# In[143]:


df['text'].head(10)


# In[144]:


df['text'].head(10).apply(text_filtering)


# # 19  Bag Of Words --->  bow

# In[145]:


from sklearn.feature_extraction.text import CountVectorizer


# In[146]:


bow_transformer = CountVectorizer(analyzer=text_filtering).fit(df['text'])


# In[147]:


bow_transformer


# In[148]:


len(bow_transformer.vocabulary_)


# In[149]:


bow_text = bow_transformer.transform(df['text'])

bow_text


# In[150]:


bow_text.nnz


# In[151]:


bow_text.shape


# In[152]:


bow_text.shape[0]


# In[153]:


bow_text.shape[1]


# In[154]:


(bow_text.nnz*100)/(bow_text.shape[0]*bow_text.shape[1])


# In[155]:


print('% of sparsity: ', (bow_text.nnz*100)/(bow_text.shape[0]*bow_text.shape[1]))
print('None Zero Numbers: ', bow_text.nnz)
print('Shape of Sparse Matrix: ', bow_text.shape)


# # 20 TF-IDF

# In[156]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[157]:


tfidf_transform = TfidfTransformer().fit(bow_text)


# In[158]:


tfidf_text = tfidf_transform.transform(bow_text)
tfidf_text


# # 21 Classifier

# In[159]:


from sklearn.naive_bayes import MultinomialNB


# In[160]:


nb = MultinomialNB()


# In[161]:


X = tfidf_text 
y = df['stars']


# In[162]:


text_rating_model = nb.fit(X,y)


# # 22

# In[163]:


predicted = text_rating_model.predict(X)


# In[164]:


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


# In[165]:


confusion_matrix(y,predicted)


# In[167]:


print('Accuracy = ', accuracy_score(y,predicted))
print('F1-score = ', f1_score(y,predicted, average='micro'))
print('Precision = ', precision_score(y,predicted, average='micro'))
print('Recall = ', recall_score(y,predicted, average='micro'))


# # 23  Split Data

# In[168]:


from sklearn.model_selection import train_test_split

text_train, text_test, label_train, label_test = train_test_split(df['text'], df['stars'], test_size=0.2, random_state=100)


# # 24 Data Pipeline

# In[170]:


from sklearn.pipeline import Pipeline


# In[171]:


pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer=text_filtering)),
        ('tfidf', TfidfTransformer()),
        ('classifier',  MultinomialNB())
])


# In[172]:


pipeline.fit(text_train, label_train)


# # 25

# In[174]:


predicted = pipeline.predict(text_test)


# In[175]:


confusion_matrix(label_test,predicted)


# In[177]:


print('Accuracy = ', accuracy_score(label_test,predicted))
print('F1-score = ', f1_score(label_test,predicted, average='micro'))
print('Precision = ', precision_score(label_test,predicted, average='micro'))
print('Recall = ', recall_score(label_test,predicted, average='micro'))


# # 26 ทำใหม่ทั้งหมด ลองแบบที่ไม่ได้ทำ  TF-IDF และไม่ได้ split test

# In[178]:


nb = MultinomialNB()


# In[179]:


X = bow_text 
y = df['stars']


# In[180]:


text_rating_model2 = nb.fit(X,y)


# In[181]:


predicted2 = text_rating_model2.predict(X)


# In[182]:


confusion_matrix(y,predicted2)


# In[184]:


print('Accuracy = ', accuracy_score(y,predicted2))
print('F1-score = ', f1_score(y,predicted2, average='micro'))
print('Precision = ', precision_score(y,predicted2, average='micro'))
print('Recall = ', recall_score(y,predicted2, average='micro'))


# In[185]:


#สรุปคือ ถ้าไม่ทำ TF-IDF จะมีความแม่นยำมากกว่า

#เพราะว่า การแยกแยะว่า คำไหนสำคัญ ไม่สำคัญ แล้วเอามาเทียบกับระดับของ Stars เป็นเรื่องที่ค่อนข้างยากและซับซ้อน


# In[ ]:




