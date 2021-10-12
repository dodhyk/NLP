# -*- coding: utf-8 -*-
"""natural_language_processing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1J-8rJQIjlpa6C3mIzAsPCQGe1T60VYjh
"""

import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.info()

test.info()

"""# Preprocessing"""

import string
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
string.punctuation

Test = 'i am a Man'
Test_token = word_tokenize(Test)
Test_token = ''.join([f for f in Test_token if f not in stopwords.words('english')])
print(Test_token)

for f in Test_token.split():
  if f not in stopwords.words('english'):
    print(f)

Test = 'Good morning beautiful people :)... I am having fun learning Machine learning and AI!!'

Test = ''.join([f for f in Test if f not in string.punctuation])
print(Test)

Test = 'nama http://t.col/hyxeohy6c'
Test = re.sub('http\S+|www.\S+', '', Test)
print(Test)

def del_punctuation(message):
  punc_remover = [text for text in message if text not in string.punctuation]
  punc_remover_join = ''.join(punc_remover)
  return punc_remover_join

def del_stopwords(message):
  stopwords_remover = [text for text in message.split() if text not in stopwords.words('english')]
  stopwords_remover_join = ' '.join(stopwords_remover)
  
  return stopwords_remover_join

def preprocessing(df):
  df = df.copy()

  # remove unused columns
  df = df.drop(['id','keyword','location'], axis=1)

  # make text into lowercase
  df['text'] = df['text'].str.lower()

  # remove punctuation
  df['text'] = df['text'].apply(del_punctuation)

  # remove url
  df['text'] = df['text'].str.replace('http\S+|www.\S+', '')

  # removing stopword
  df['text'] = df['text'].apply(del_stopwords)

  return df

x = preprocessing(train)

X_test = preprocessing(test)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
tweets_countvectorizer = vectorizer.fit_transform(x['text']).toarray()

x_test = vectorizer.transform(X_test['text']).toarray()

X_train = tweets_countvectorizer
y_train = x['target']

X_train.shape

x_test.shape

from sklearn.svm import SVC

model_SVC = SVC()
model_SVC.fit(X_train, y_train)

y_predict = model_SVC.predict(x_test)

data = {'id':test['id'],
        'target':y_predict}

df = pd.DataFrame(data)

df.to_csv('submission.csv', index=False)

