# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 16:19:13 2018

@author: kevin
"""

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
from matplotlib import pyplot as plot
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

newsSarc = pd.read_json("newsHeadline/Sarcasm_Headlines_Dataset.json", dtype=False, lines=True)
type(newsSarc)


newsSarc.dropna(subset=['headline'], inplace=True)
print(newsSarc.columns, newsSarc.shape)
#there are no missing headlines

#how many headlines are sarcastic vs. not? 
newsSarc['is_sarcastic'].hist()

#split the df into training and validation parts
train_texts, valid_texts, y_train, y_valid = \
        train_test_split(newsSarc['headline'], newsSarc['is_sarcastic'], random_state=10)

print(train_texts.shape, valid_texts.shape, y_train.shape, y_valid.shape)

train_head_small = train_texts.sample(300, random_state=10)
y_train_small = y_train.sample(300, random_state=10)
valid_head_small = valid_texts.sample(300, random_state=10)
y_valid_small = y_valid.sample(300, random_state=10)

#set up tf idf
tf_idf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2, lowercase=True)
# multinomial logistic regression a.k.a softmax classifier
logit = LogisticRegression(C=1, n_jobs=4, solver='lbfgs', 
                           random_state=10, verbose=1)
# sklearn's pipeline
# First do tf_idf, then do logistic regression
tfidf_logit_pipeline = Pipeline([('tf_idf', tf_idf), 
                                 ('logit', logit)])

#fit the model
tfidf_logit_pipeline.fit(train_head_small, y_train_small)
trans = tf_idf.fit_transform(train_head_small)
valid_pred = tfidf_logit_pipeline.predict(valid_head_small)
accuracy_score(y_valid_small, valid_pred)
conf = confusion_matrix(y_valid_small, valid_pred)

#make more sense of the TFidf results (source is https://buhrmann.github.io/tfidf-analysis.html)
def class_report(conf_mat):
    tp, fp, fn, tn = conf_mat.flatten()
    measures = {}
    measures['accuracy'] = (tp + tn) / (tp + fp + fn + tn)
    measures['specificity'] = tn / (tn + fp)        # (true negative rate)
    measures['sensitivity'] = tp / (tp + fn)        # (recall, true positive rate)
    measures['precision'] = tp / (tp + fp)
    measures['f1score'] = 2*tp / (2*tp + fp + fn)
    return measures

class_report(conf)


results1 = list(tf_idf.vocabulary_.items())
results2 = np.array(trans[np.nonzero(trans.todense())])
print(trans)

#show top terms (source https://buhrmann.github.io/tfidf-analysis.html)
Xtr = tf_idf.fit_transform(train_head_small)
features = tf_idf.get_feature_names()
def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    ''' Top tfidf features in specific document (matrix row) '''
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

print(Xtr.shape)
top_feats_in_doc(Xtr, features, 99)