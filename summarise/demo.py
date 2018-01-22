#Actual demo that uses the text summariser module to summarise text into sentences
from nltk import sent_tokenize
from nltk.corpus import inaugural
from TextSummariserDemo import TextSummariser as summarise


#Summarising Obama Speech
sample=inaugural.raw("2009-Obama.txt")
text=str(sample).strip()
thesummarised=summarise.FrequencySummariser().summarize(text,1)
print (thesummarised)

#summarising entire body of work by Shakespeare
textandanswers=""
textpath = '/home/iamukasa/PycharmProjects/NLP/CustomCopora/analyze.txt'
Alltext=open(textpath, "r").read()
for a in Alltext.split('WNDEF'):
    textandanswers +=a
thesummarised=summarise.FrequencySummariser().summarize(textandanswers.strip(),1)
print (thesummarised)
