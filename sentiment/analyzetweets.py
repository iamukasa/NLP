#Access and authorize our Twitter credentials from credentials.py
import tweepy
from sentiment.credentials import *
import sentiment.sentimentmodule as mod

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

B="sex dolls"
new_tweets = api.search(B,2000)

# list of specific strings we want to check for in Tweets


for s in new_tweets:

            sn = s.user.screen_name
            # sentiment =mod.sentiment(s.text)
            sentiment,confidence =mod.sentiment(s.text)

            print(s.text,sentiment,confidence)
            logffiles = open("/home/iamukasa/PycharmProjects/NLP/sentialgos/slog.txt", "a")
            logffiles.write(sentiment)
            logffiles.write('\n')
            logffiles.close()





