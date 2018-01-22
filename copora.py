from nltk.corpus import inaugural, twitter_samples
from nltk.tokenize import sent_tokenize
# reading files from nltk copora

sample=inaugural.raw("2009-Obama.txt")
tok=sent_tokenize(sample)
print(tok[25:30])

tweet_pos=twitter_samples.raw("/home/iamukasa/nltk_data/corpora/twitter_samples/positive_tweets.json")
tweet_neg=twitter_samples.raw("/home/iamukasa/nltk_data/corpora/twitter_samples/negative_tweets.json")
print(tweet_neg)
print(tweet_pos)

# reading custm data
textpath = '/home/iamukasa/Downloads/analyze.txt'
Alltext=open(textpath, "r").read()
print(Alltext)
