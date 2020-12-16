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


df = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t')
df


# # 2

# In[3]:


df.head(10)


# In[4]:


df.tail(10)


# In[5]:


df.sample(10)


# # 3

# In[6]:


df.info()


# In[7]:


df.describe()


# # 4

# In[8]:


sns.countplot(df['Liked'])


# # 5

# In[9]:


df['Review'].apply(len)


# In[10]:


df['Length'] = df['Review'].apply(len)


# In[11]:


df


# # 6

# In[12]:


df['Length'].plot(bins=100, kind='hist')


# # 7

# In[13]:


df.hist(column='Length', by='Liked', bins=100, figsize=(14,8))


# # 8

# In[14]:


df['Length'].max()


# In[15]:


df[df['Length']==149]


# In[16]:


df[df['Length']==149]['Review']


# In[17]:


df[df['Length']==149]['Review'].iloc[0]


# # 9

# In[18]:


df.groupby('Liked').mean()


# In[19]:


df.Length.describe()


# # 10

# In[20]:


for i in range(10):
    print('\n', df[df['Liked']==0]['Review'].iloc[i])


# In[21]:


for i in range(10):
    print('\n', df[df['Liked']==1]['Review'].iloc[i])


# # 11

# In[27]:


import string
from nltk.corpus import stopwords


# In[28]:


import nltk
nltk.download('stopwords')


# In[29]:


def text_filtering(review_text):
    #ลบ punctuation
    no_punc = [x for x in review_text if x not in string.punctuation]
    
    #นำมารวมกันใหม่ ใช้ join
    no_punc = ''.join(no_punc)
    
    #ลบ stopwords
    text = [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]
    
    return text


# In[30]:


df['Review'].head(10)


# In[31]:


df['Review'].head(10).apply(text_filtering)


# # 12

# In[32]:


from sklearn.feature_extraction.text import CountVectorizer


# In[33]:


bow_transformer = CountVectorizer(analyzer=text_filtering).fit(df['Review'])


# In[34]:


bow_transformer


# # 13

# In[35]:


len(bow_transformer.vocabulary_)


# # 14

# In[36]:


bow_review = bow_transformer.transform(df['Review'])


# In[37]:


bow_review


# # 15

# In[38]:


bow_review.nnz


# In[42]:


bow_review.shape


# In[43]:


bow_review.shape[0]


# In[44]:


bow_review.shape[1]


# In[47]:


#อันนี้คือวิธีที่ถูกต้อง

(bow_review.nnz*100)/(bow_review.shape[0]*bow_review.shape[1])


# In[48]:


#การเขียนโปรแกรม ไม่ควรใช้ค่าคงตัว

(bow_review.nnz*100)/(1000*2159)


# In[50]:


print('% of sparsity: ', (bow_review.nnz*100)/(bow_review.shape[0]*bow_review.shape[1]))
print('None Zero Numbers: ', bow_review.nnz)
print('Shape of Sparse Matrix: ', bow_review.shape)


# # 16

# In[51]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[52]:


tfidf_transform = TfidfTransformer().fit(bow_review)


# In[53]:


tfidf_review = tfidf_transform.transform(bow_review)
tfidf_review


# In[55]:


#ทดสอบลองดูเฉยๆ เพื่อให้เข้าใจมากขึ้น จริงๆเสร็จตั้งแต่ 3 บรรทัดแรกแล้ว


# In[56]:


df['Review'][3]


# In[58]:


test_text = text_filtering(df['Review'][3])
test_text


# In[61]:


bow_test = bow_transformer.transform([df['Review'][3]])
bow_test


# In[62]:


print(bow_test)


# In[63]:


tfidf_test = tfidf_transform.transform(bow_test)


# In[64]:


print(tfidf_test)


# # 17

# In[65]:


from sklearn.naive_bayes import MultinomialNB


# In[66]:


nb = MultinomialNB()


# In[67]:


X = tfidf_review 
y = df['Liked']


# In[68]:


review_rating_model = nb.fit(X,y)


# In[69]:


predicted = review_rating_model.predict(X)


# # 18

# In[70]:


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


# In[74]:


confusion_matrix(y,predicted)


# # 19

# In[75]:


print('Accuracy = ', accuracy_score(y,predicted))
print('F1-score = ', f1_score(y,predicted))
print('Precision = ', precision_score(y,predicted))
print('Recall = ', recall_score(y,predicted))


# # 20  GaussianNB

# In[76]:


from sklearn.naive_bayes import GaussianNB


# In[77]:


nb = GaussianNB()


# In[78]:


review_rating = nb.fit(X.todense(),y)


# In[79]:


predicted = review_rating.predict(X.todense())


# # 21

# In[80]:


confusion_matrix(y,predicted)


# # 22

# In[81]:


print('Accuracy = ', accuracy_score(y,predicted))
print('F1-score = ', f1_score(y,predicted))
print('Precision = ', precision_score(y,predicted))
print('Recall = ', recall_score(y,predicted))


# # 23

# In[82]:


from sklearn.model_selection import train_test_split

review_train, review_test, label_train, label_test = train_test_split(df['Review'], df['Liked'], test_size=0.2, random_state=100)


# # 24

# In[83]:


from sklearn.pipeline import Pipeline


# In[85]:


pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer=text_filtering)),
        ('tfidf', TfidfTransformer()),
        ('classifier',  MultinomialNB())
])


# In[86]:


pipeline.fit(review_train, label_train)


# In[87]:


predicted = pipeline.predict(review_test)


# # 25

# In[88]:


confusion_matrix(label_test,predicted)


# # 26

# In[89]:


print('Accuracy = ', accuracy_score(label_test,predicted))
print('F1-score = ', f1_score(label_test,predicted))
print('Precision = ', precision_score(label_test,predicted))
print('Recall = ', recall_score(label_test,predicted))


# In[ ]:




