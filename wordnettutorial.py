#demonstrating the powerful wordnet feature

from nltk.corpus import wordnet
from pip._vendor.distlib.compat import raw_input

# str=raw_input("pick a word :")
# syns = wordnet.synsets(str)

syns=wordnet.synsets("give")


# synsnet
print(syns[0].name())

# just the word
print(syns[0].lemmas()[0].name())

# Definitiion
print(syns[0].definition())

# examples
print(syns[4].definition())
print(syns[4].examples())

synonyms = []
antonyms = []

#Fetching synonyms and antonyms of the word
for syns in wordnet.synsets("good"):
    for l in syns.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))

w1 = wordnet.synset("bot.n.01")
w2 = wordnet.synset("robot.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset("ship.v.01")
w2 = wordnet.synset("ferry.n.01")
print(w1.wup_similarity(w2))










