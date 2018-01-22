from string import punctuation

import nltk
from nltk.corpus import stopwords, inaugural
from nltk.tokenize import sent_tokenize, word_tokenize
from  nltk.stem import PorterStemmer


#Perfoming different nltk preprocessing of your data

#Data We will use
sample=inaugural.raw("2009-Obama.txt")


#Tokenizing : Breaking down the body of text
print(sent_tokenize(sample))
#Sentence Tokenizing: Breaking down by sentence
print(word_tokenize(sample))
#Word Tokenizing:Breaking down body of text by words


#STOP WORDS: removing grammar and prepositions that add no meaning to data
stop_words=set(stopwords.words('english'))
print(stop_words)
stop_words=set(stopwords.words('english')+
                            list(punctuation)+
                            [u"'s",'""'])
print(stop_words)

#removing stop words from copora
allwords=[]
for w in word_tokenize(sample):
    if w not  in stop_words:
        allwords.append(w)

print(allwords)



#stemming Getting the root of the word
stemmed_words=[]
ps=PorterStemmer()

for w in allwords:
    stemmed_words.append(ps.stem(w))

print(stemmed_words)


#part of speech tagging: recognising nouns verbs and other parts of speech
tagged=nltk.pos_tag(stemmed_words)
print(tagged)


#named entity recognition
namedEnt = nltk.ne_chunk(tagged)
print(namedEnt)


#lemattizing

lemmatizzer= nltk.WordNetLemmatizer()

lemmattized_words=[]
for w in stemmed_words:
    lemmattized_words.append(lemmatizzer.lemmatize(w))
print(lemmattized_words)








