#Using An Uusupervised AI model to identify topic on the sophie bot training data
from string import punctuation

import nltk
from nltk.corpus import stopwords, inaugural
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Importing CustomCorpora Data
textandanswers=""
textpath = '/home/iamukasa/PycharmProjects/NLP/data/analyze.txt'
Alltext=open(textpath, "r").read()
C=Alltext.splitlines()
print(C)

# #Importing Obama Speech
# sample=inaugural.raw("2009-Obama.txt")
# C=nltk.sent_tokenize(sample)
# print(C)



vectorizer=TfidfVectorizer(max_df=0.5,stop_words='english')
X=vectorizer.fit_transform(C)

km=KMeans(n_clusters=10,init='k-means++',max_iter=100,n_init=1,verbose=True)
km.fit(X)
centers=km.cluster_centers_
labels=km.labels_

vocabulary=vectorizer.vocabulary_
fnames=vectorizer.get_feature_names()

# print(labels)
# print(centers)
print(vocabulary)
# print(fnames)
print(vectorizer.inverse_transform(centers))