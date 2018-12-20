# -*- coding: utf-8 -*-
########
#some eda from https://www.kaggle.com/danofer/loading-sarcasm-data
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
print(check_output(["ls", "C:/Users/kevin/Desktop/Career Prep/Kevin work Portfolio/Personal Project/sarcasm"]).decode("utf8"))

train = pd.read_csv("C:/Users/kevin/Desktop/Career Prep/Kevin work Portfolio/Personal Project/sarcasm/train-balanced-sarcasm.csv")
print(train.shape)
print(train.columns)

#drop rows that have missing comments
train.dropna(subset=['comment'], inplace=True)

# Parse UNIX epoch timestamp as datetime: 
# df.created_utc = pd.to_datetime(df.created_utc,unit="s") # Applies to original data , which had UNIX Epoch timestamp! 
train.created_utc = pd.to_datetime(train.created_utc,infer_datetime_format=True) # Applies to original data , which had UNIX Epoch timestamp! 

train.describe()

########
train['label'].hist() # 50% od každé super :)

##see a sample of comments
train['comment'].sample(10)
train[train.label == 1]["comment"].sample(10).tolist()


#how many comments are in each subreddit?
train.groupby(["subreddit"]).count()["comment"].sort_values()
#learn more about the subreddits and the frequency of sarcastic labels
sub_df = train.groupby('subreddit')['label'].agg([np.size, np.mean, np.sum])
sub_df.sort_values(by='sum', ascending=False).head(10)
sub_df[sub_df['size'] > 1000].sort_values(by='mean', ascending=False).head(10)


#does score provide any indicator of a sarcastic comment?
sub_df_sc = train[train['score'] < 0].groupby('score')['label'].agg([np.size, np.mean, np.sum])
sub_df_sc[sub_df_sc['size'] > 300].sort_values(by='mean', ascending=False).head(30)
sub_df_sc[sub_df_sc['size'] > 300].sort_values(by='mean', ascending=False).head(30)

#split the df into training and validation parts
train_texts, valid_texts, y_train, y_valid = \
        train_test_split(train['comment'], train['label'], random_state=17)
        
print(train_texts.shape, valid_texts.shape, y_train.shape, y_valid.shape)

train_texts_small = train_texts.sample(300, random_state=17)
y_train_small = y_train.sample(300, random_state=17)
valid_texts_small = valid_texts.sample(300, random_state=17)
y_valid_small = y_valid.sample(300, random_state=17)

#start TFidf (source https://www.kaggle.com/kashnitsky/topic-4-practice-sarcasm-detection-solution)
# build bigrams, put a limit on maximal number of features (max # of words?)
# and minimal word frequency (min_df)
'''
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2, lowercase=True)
scores = vectorizer.fit_transform(train_texts_small)
scoreMat = pandas.DataFrame(vectorizer.get_feature_names()).append(pandas.DataFrame(scores.toarray()))
print(scoreMat.shape)
print(vectorizer.get_feature_names())
print(scores.toarray())
'''

tf_idf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2, lowercase=True)
# multinomial logistic regression a.k.a softmax classifier
logit = LogisticRegression(C=1, n_jobs=4, solver='lbfgs', 
                           random_state=17, verbose=1)
# sklearn's pipeline
# First do tf_idf, then do logistic regression
tfidf_logit_pipeline = Pipeline([('tf_idf', tf_idf), 
                                 ('logit', logit)])

#fit the model
tfidf_logit_pipeline.fit(train_texts_small, y_train_small)
trans = tf_idf.fit_transform(train_texts_small)
valid_pred = tfidf_logit_pipeline.predict(valid_texts_small)
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


#show weights
results1 = list(tf_idf.vocabulary_.items())
results2 = np.array(trans[np.nonzero(trans.todense())])
print(trans)

#show top terms (source https://buhrmann.github.io/tfidf-analysis.html)
Xtr = tf_idf.fit_transform(train_texts_small)
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