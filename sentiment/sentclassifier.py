import nltk
import random
# from nltk.corpus import movie_reviews
from nltk.corpus import twitter_samples
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from nltk.classify import ClassifierI
from statistics import  mode
from nltk.corpus import TwitterCorpusReader
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC


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


tweet_pos=twitter_samples.raw("/home/iamukasa/nltk_data/corpora/twitter_samples/positive_tweets.json")
tweet_neg=twitter_samples.raw("/home/iamukasa/nltk_data/corpora/twitter_samples/negative_tweets.json")

documents=[]

for t in tweet_pos.split("\n"):
    a=(t,"pos")
    documents.append((t,"pos"))

for t in tweet_neg.split("\n"):
    a=(t,"neg")
    documents.append(a)



 #Pickle Documents
save_documents=open("/home/iamukasa/PycharmProjects/NLP/sentialgos/documents.pickle","wb")
pickle._dump(  documents,save_documents)
save_documents.close()


all_words=[]
tweet_pos_words= nltk.word_tokenize(tweet_pos)
tweet_neg_words= nltk.word_tokenize(tweet_neg)

for w in tweet_pos_words:
    all_words.append(w.lower())

for w in tweet_neg_words:
    all_words.append(w.lower())



all_words=nltk.FreqDist(all_words)

words_features=list(all_words.keys())[:4000]


#picklewordfeatures
save_words_feature=open("/home/iamukasa/PycharmProjects/NLP/sentialgos/words_feature.pickle","wb")
pickle._dump(words_features,save_words_feature)
save_words_feature.close()


def find_features(documents):
    # words=nltk.word_tokenize(documents)
    words=set(documents)
    features={}
    for w in words_features:
        features[w]=(w in words)

    return features


featuresets=[(find_features(rev),category) for (rev,category) in documents]
random.shuffle(featuresets)


#pickle featuresets
save_featuresets=open("/home/iamukasa/PycharmProjects/NLP/sentialgos/featuresets.pickle","wb")
pickle._dump(featuresets,save_featuresets)
save_featuresets.close()



training_set=featuresets[:2000]
testing_set=featuresets[2000:]

#pickle training set
save_training_set=open("/home/iamukasa/PycharmProjects/NLP/sentialgos/training_set.pickle","wb")
pickle._dump(training_set,save_training_set)
save_training_set.close()

#pickle testing set
save_testing_set=open("/home/iamukasa/PycharmProjects/NLP/sentialgos/testing_set.pickle","wb")
pickle._dump(testing_set,save_testing_set)
save_testing_set.close()

original_classifier=nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes Accuracy:",(nltk.classify.accuracy(original_classifier,testing_set)*100))
original_classifier.show_most_informative_features(30)

#pickle classifier
save_classifier=open("/home/iamukasa/PycharmProjects/NLP/sentialgos/naivebayes.pickle","wb")
pickle._dump(original_classifier,save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

save_classifier = open("/home/iamukasa/PycharmProjects/NLP/sentialgos/MNB_classifier5k.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

save_classifier = open("/home/iamukasa/PycharmProjects/NLP/sentialgos/BernoulliNB_classifier5k.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

save_classifier = open("/home/iamukasa/PycharmProjects/NLP/sentialgos/LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

save_classifier = open("/home/iamukasa/PycharmProjects/NLP/sentialgos/LinearSVC_classifier5k.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()


##NuSVC_classifier = SklearnClassifier(NuSVC())
##NuSVC_classifier.train(training_set)
##print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
print("SGDClassifier accuracy percent:",nltk.classify.accuracy(SGDC_classifier, testing_set)*100)

save_classifier = open("/home/iamukasa/PycharmProjects/NLP/sentialgos/SGDC_classifier5k.pickle","wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()





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



print("Classification:",
      voted_classifier.classify(testing_set[1][0]),
      "confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:",
      voted_classifier.classify(testing_set[2][0]),
      "confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:",
      voted_classifier.classify(testing_set[3][0]),
      "confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:",
      voted_classifier.classify(testing_set[4][0]),
      "confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:",
      voted_classifier.classify(testing_set[5][0]),
      "confidence %:",voted_classifier.confidence(testing_set[0][0])*100)



def sentiment(text):
    feats=find_features(text)
    return voted_classifier.classify(feats)




