'''
Created on Nov 9, 2013
@author: Rahul
'''

#*************All Imports*************
import re
import csv
import nltk
import itertools
import sys
from nltk.corpus import stopwords
from nltk.util import tokenwrap
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
#*************************************

       
#read Contractions Data and save it in contractionDict
contractionDict = {} 
contractionsData = csv.reader(open('Data/Contractions.csv', 'r' ))
for line in contractionsData:
    replacement = line[1]
    contractiontext = line[0]
    contractionDict[contractiontext] = replacement
contractions_re = re.compile('(%s)' % '|'.join(contractionDict.keys()))
#end  


#method to replace contractions in given text with text from the contractionDict
def replace_Contractions(s, contractions_dict=contractionDict):
    def replace(match):
        return contractionDict[match.group(0)]
    return contractions_re.sub(replace, s)
#end


#start process_tweet
def processTweet(tweet):
    # process the tweets
 
    #Convert to LowerCase
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[\s]+)|(htt[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces i.e replace 2 white spaces with one
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    #replace Contractions
    tweet = replace_Contractions(tweet)
    
    return tweet
#end

    
#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end


#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords file and build a list
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')
    
    st = open(stopWordListFileName, 'r')
    line = st.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = st.readline()
    st.close()
    return stopWords
#end    


#start getfeatureVector
def getFeatureVector(tweet, stopWordsList):
    featureVector = []
    bigrams = []
    #split tweet into words
    words = tweet.split()
    print
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopWordsList or val is None):
            continue
        else:
            w = lemma_process(w)
            sys.stdout.write("After: ",w,"\n")    
            featureVector.append(w.lower())
    return featureVector
#end    


#lemmatizer
def lemma_process(w):
    listGrammer = []
    lemmatzr = WordNetLemmatizer()
    listGrammer=nltk.pos_tag(w)
    grammerVal = listGrammer.pop()[1]
    
    if(grammerVal == 'JJ'):
        w = lemmatzr.lemmatize(w,wordnet.ADJ)
    elif(grammerVal == 'VB'):
        w = lemmatzr.lemmatize(w,wordnet.VERB)
    elif(grammerVal == 'NN'):
        w = lemmatzr.lemmatize(w,wordnet.NOUN)
    elif(grammerVal == 'RB'):
        w = lemmatzr.lemmatize(w,wordnet.ADV)
    else:
        w = lemmatzr.lemmatize(w,wordnet.NOUN)
    return w
#end


#get SparseMatrix
def getSparseMatrix(tweets, featureList):
    sparseMatrix = []
    
    # loop for building the sparseMatrix and TrainSentiment List of the Training data
    for tweet in tweets:
        tweetText = tweet[0]
        featureVector = []
        
        #constructs the feature Vector 
        for word in featureList:
            if word in tweetText:
                featureVector.append(1)
            else:
                featureVector.append(0)
                
        #appending the constructed feature vector to the Sparse Matrix                    
        sparseMatrix.append(featureVector)
    
    return sparseMatrix
#end




#get Sentiment List
def getSentimentList(tweets):
    trainSentiment = []
    
    # loop for building the sparseMatrix and TrainSentiment List of the Training data
    for tweet in tweets:
        tweetSentiment = tweet[1]

        #builds  the sentiment Vector
        if(tweetSentiment == "Cessation"):
            trainSentiment.append(1)
        else:
            trainSentiment.append(0)
            
    return trainSentiment
#end






