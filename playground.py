import json


from nltk.corpus import twitter_samples
from pip._vendor.distlib.compat import raw_input

from sentiment import sentimentmodule as s



i=0
while i >10000000:
    str=raw_input("enter your text :")
    print(s.sentiment(str))
    i =i+1


