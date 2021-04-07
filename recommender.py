#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[31]:


book=pd.read_csv("D:\\projects\\recommendation\\Book.csv",encoding="ISO-8859-1")


# In[3]:


book.head()


# In[4]:


book.shape


# In[5]:


book.columns


# In[6]:


book['Book.Rating']


# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[8]:


tfidf=TfidfVectorizer(stop_words="english")


# In[9]:


book['Book.Rating'].isnull().sum()


# In[10]:


tfidf_matrix=tfidf.fit_transform(book['Book.Title'])


# In[12]:


tfidf_matrix.shape


# In[11]:


from sklearn.metrics.pairwise import linear_kernel


# In[13]:


cosine_sim_matrix=linear_kernel(tfidf_matrix,tfidf_matrix)


# In[20]:


book_index=pd.Series(book.index,index=book['Book.Title']).drop_duplicates()


# In[32]:


book_index['Springs in the Valley']


# In[37]:


book_df=book.rename({'User.ID': 'ID', 'Book.Title': 'Name', 'Book.Rating':'Rating'}, axis=1, inplace=True)


# In[43]:


def get_books(Name,Rating):
    book_id=book_index[Name]
    cosine_scores=list(enumerate(cosine_sim_matrix[book_id]))
    cosine_scores=sorted(cosine_scores,key=lambda x:x[1],reverse=True)
    cosine_scores_10=cosine_scores[0:Rating+1]
    
    book_idx = [i[0] for i in cosine_scores_10]
    book_scores=[i[1] for i in cosine_scores_10]
    
    book_similar=pd.DataFrame(columns=["Name","Rating"])
    book_similar["Name"]=book.loc[book_idx,"Name"]
    book_similar["Rating"]=book_scores
    book_similar.reset_index(inplace=True)
   # book_similar.drop(["ID"],axis=1,inplace=True)
    print(book_similar)


# In[44]:


get_books("The Middle Stories",Rating=6)


# In[ ]:




