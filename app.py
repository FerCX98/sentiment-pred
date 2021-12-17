#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from flask import Flask,render_template,url_for,request


app = Flask(__name__)
@app.route('/')
def home():
	return render_template('home.html')
    
@app.route('/predict',methods=['POST'])
def predict():
    data1 = pd.read_csv("negative.csv")
    data2 = pd.read_csv("positive.csv")
    data = pd.concat([data1,data2],ignore_index=True)
    print("3 or less star examples: ")
    print(len(data1))
    print("4 or 5 star examples: ")
    print(len(data2))
    print("both: ")
    print(len(data))


    # In[3]:


    data1.describe()


    # In[4]:


    data2.describe()


    # In[5]:


    data.describe()


    # In[6]:


    dX=[]
    dY=[]
    for c in range(len(data)):
        dX.append(data.loc[c,"Review"])
        dY.append(data.loc[c,"Label"])


    # In[ ]:





    # In[7]:


    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(dX)
    X_train_counts.shape


    # In[8]:


    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_train_tfidf.shape


    # In[9]:


    clf = LogisticRegression(max_iter=50000).fit(X_train_tfidf, dY)
    pred=clf.predict(X_train_tfidf)
    np.mean(pred==dY)


    # In[10]:


    import joblib
    joblib.dump(clf, 'NB_spam_model.pkl')
    
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = count_vect.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)
    

if __name__ == '__main__':
    app.run(debug=True)

