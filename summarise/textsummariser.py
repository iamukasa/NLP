#module to summarize text

from string import punctuation

import nltk
from nltk.corpus import stopwords
from nltk import defaultdict
import heapq
import sys


class FrequencySummariser:
    def __init__(self,min_cut=0.1,max_cut=0.9):
        self.min_cut=min_cut
        self.max_cut=max_cut
        self._stopwords=set(stopwords.words('english')+
                            list(punctuation)+
                            [u"'s",'""'])
    def _compute_frequencies(self,word_sent,customStopWords=None):
        freq= defaultdict(int)
        if customStopWords is None:
            stopwords=set(self._stopwords)
        else:
            stopwords=set(customStopWords).union(self._stopwords)
        for sentence in word_sent:
            for word in sentence:
                if word not in stopwords:
                    freq[word] +=1
        m=float(max(freq.values()))
        for word in list(freq):
            freq[word]=freq[word]/m
            if freq[word] >= self.max_cut or freq[word]<=self.min_cut:
                del freq[word]
        return freq

    def summarize(self,article,n):
        answers=""
        sentences= nltk.sent_tokenize(article)
        word_sent=[nltk.word_tokenize(s.lower()) for s in sentences]
        self._freq=self._compute_frequencies(word_sent)
        ranking= nltk.defaultdict(int)
        for i,sentence in enumerate(word_sent):
            for word in sentence:
                if word in self._freq:
                    ranking[i]+=self._freq[word]


        sentences_index=heapq.nlargest(n,ranking,key=ranking.get)

        for j in sentences_index:

            answers+=" "+sentences[j]

        return answers


