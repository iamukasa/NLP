import nltk
import random
# from nltk.corpus import movie_reviews
from nltk.corpus import twitter_samples
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from nltk.classify import ClassifierI
from statistics import  mode

class VoteClassifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers=classifiers

    def classify(self, features):
        votes=[];
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
            return mode(votes)

    def confidence(self,features):
        votes=[]
        for c in self._classifiers:
            v= c.classify(features)
            votes.append(v)

        choice_votes=votes.count(mode(votes))
        conf=choice_votes/len(votes)
        return conf

#fetchdocuments
documents_f=open("/home/iamukasa/PycharmProjects/NLP/sentialgos/documents.pickle","rb")
documents=pickle.load(documents_f)
documents_f.close()



#fetch word_features
word_feature_f=open("/home/iamukasa/PycharmProjects/NLP/sentialgos/words_feature.pickle","rb")
word_feature=pickle.load(word_feature_f)
word_feature_f.close()

def find_features(documents):
    words=nltk.word_tokenize(documents)
    features={}
    for w in word_feature:
        features[w]=(w in words)
        print(features)

    return features


#fetch featuresets
featuresets_f=open("/home/iamukasa/PycharmProjects/NLP/sentialgos/featuresets.pickle","rb")
featuresets=pickle.load(featuresets_f)
featuresets_f.close()

#training set
training_set_f=open("/home/iamukasa/PycharmProjects/NLP/sentialgos/training_set.pickle","rb")
training_set=pickle.load(training_set_f)
training_set_f.close()

#Testing Set
testing_set_f=open("/home/iamukasa/PycharmProjects/NLP/sentialgos/testing_set.pickle","rb")
testing_set=pickle.load(testing_set_f)
testing_set_f.close()


#Naive Bayes Classifier
naivebayesclassifier_f=open("/home/iamukasa/PycharmProjects/NLP/sentialgos/naivebayes.pickle","rb")
original_classifier=pickle.load(naivebayesclassifier_f)
naivebayesclassifier_f.close()

saveMNB_classifier = open("/home/iamukasa/PycharmProjects/NLP/sentialgos/MNB_classifier5k.pickle","rb")
MNB_classifier=pickle.load(saveMNB_classifier)
saveMNB_classifier.close()

saveBernoulliNB_classifier = open("/home/iamukasa/PycharmProjects/NLP/sentialgos/BernoulliNB_classifier5k.pickle","rb")
BernoulliNB_classifier=pickle.load(saveBernoulliNB_classifier)
saveBernoulliNB_classifier.close()


saveLogisticRegression_classifier = open("/home/iamukasa/PycharmProjects/NLP/sentialgos/LogisticRegression_classifier5k.pickle","rb")
LogisticRegression_classifier=pickle.load(saveLogisticRegression_classifier)
saveLogisticRegression_classifier.close()


saveLinearSVC_classifier = open("/home/iamukasa/PycharmProjects/NLP/sentialgos/LinearSVC_classifier5k.pickle","rb")
LinearSVC_classifier=pickle.load(saveLinearSVC_classifier)
saveLinearSVC_classifier.close()


saveSGDC_classifier = open("/home/iamukasa/PycharmProjects/NLP/sentialgos/SGDC_classifier5k.pickle","rb")
SGDC_classifier=pickle.load(saveSGDC_classifier)
saveSGDC_classifier.close()




voted_classifier = VoteClassifier(
                                  original_classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  SGDC_classifier,
                                  LogisticRegression_classifier)


print("voted_classifier:",(nltk.classify.accuracy(voted_classifier,testing_set)*100))

print("Classification:",
      voted_classifier.classify(testing_set[0][0]),
      "confidence %:",voted_classifier.confidence(testing_set[0][0]))


def sentiment(text):
    feats=find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)

