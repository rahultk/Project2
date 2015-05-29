'''
Created on Nov 9, 2013
@author: Rahul
'''

#*************All Imports*************
#import regex
# Added simple comment to check GitHub access
import nltk 
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import stopwords

import itertools
import time
import sys

import re
import datetime
import collections
import numpy as np
from FormatText import *

from sklearn import svm
from sklearn import naive_bayes
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from nltk.featstruct import FeatList
#*************************************

#initialize stopWords and adding the necessary extra stop-words
stopWords = []
stopWords = stopwords.words('english')
stopWords.extend(["url","at_user"])

    
def main():
    start_time = time.time()
    print "Inside main"
    
    #creating file object for reading the text from csv file
    data = csv.reader(open('TrainData/SmDataWORetweets.csv', 'r' ))
    
    #enter the number of folds for Cross Validation
    num_folds = 5
    
    #reading from file and creating a List[] of total Data
    totalData = []
    for record in data:
        totalData.append(record)
        
    #size of each partition    
    subset_size = len(totalData)/num_folds    

    #keep total accuracy
    totalAccuracy = 0


    #the actual cross validation loop
    for i in range(num_folds):
        testing_this_round = totalData[i * subset_size:][:subset_size]
#         sys.stdout.write("TestData Range : ")
#         sys.stdout.write(str())
        training_this_round = totalData[:i*subset_size]+totalData[(i+1)*subset_size:]
        accuracy = classifyScikit_SVM(training_this_round, testing_this_round) 
        totalAccuracy = totalAccuracy + accuracy
    #end of Cross Validation for loop
    
    
    #Finale Accuracy
    finalAccuracy = totalAccuracy/num_folds
    sys.stdout.write("\n\n***** Final Accuracy = ")
    sys.stdout.write(str(finalAccuracy))
    print "****"
   
   
    #Printing total program Time
    sys.stdout.write("\n Total Program run time : ")
    print time.time() - start_time, "seconds"
    sys.stdout.write("Number of Folds : ")
    print str(num_folds)
   
    
#end of Main        



#Classify the data given using train and test data 
def classifyNltk_NB(rawTrainData, rawTestData):
    
    #Build tweets array 
    tweets = [] 
    for line in rawTrainData:
        sentiment = line[1]
        tweetText = line[0]
        processedTweet = processTweet(tweetText)
        featureVector = getFeatureVector(processedTweet, stopWords)
        tweets.append((featureVector, sentiment));  
    #end loop


    #Printing Tweets
    print "\n\nPrint tweets : "
    for x in tweets:
        print x
    #end printing tweets


    #loop for building FeatureList
    featureList = []    
    for tweet in tweets:    
        for word in tweet[0]:
            if(word in featureList):
                continue
            else:
                featureList.append(word.lower())
    #end of Loop for FeatureList Build


    #gets feature vector from the tweetText
    def extract_features(tweet):
        tweet_words = set(tweet)
        features = {}
        for word in featureList:
            features['contains(%s)' % word] = int(word in tweet_words)
        return features
    #end


    #the training set corresponds to the Sparse matrix
    training_set = nltk.classify.util.apply_features(extract_features, tweets)
    
    #using the Sparse Matrix(training_set) to train the Classifier(NBClassifier)
    NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

    #Test the classifier
    testData = []


    # loop to build Set-Up testData list which is similar to tweets List and is used to build Sparse Matrix for Test Data 
    for testTweet in rawTestData:
        testTweetText = testTweet[0]
        sentimentTestTweet = testTweet[1]
        processedTestTweet = processTweet(testTweetText)
        testTweetFeature = getFeatureVector(processedTestTweet, stopWords)
        testData.append((testTweetFeature, sentimentTestTweet));
    #print NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet, stopWords)))


    #the test_set corresponds to the Sparse matrix of Test Data
    test_set = nltk.classify.util.apply_features(extract_features, testData)
    
    
    print test_set
    accu =  nltk.classify.accuracy(NBClassifier, test_set) 
    print "\n Accuracy : "
    print accu
    
    #return the Accuracy of the NBClassifier for the given Training and Test Data    
    return accu
#end of classify_NB



#Classify the data given using train and test data 
def classifyScikit_SVM(rawTrainData, rawTestData):
    
    
    #Build tweets array 
    tweets = []
    print "\n\n** Print tweets **" 
    for line in rawTrainData:
        sentiment = line[1]
        tweetText = line[0]
        processedTweet = processTweet(tweetText)
        featureVector = getFeatureVector(processedTweet, stopWords)
        tweets.append((featureVector, sentiment))          
        print ((featureVector, sentiment)),
    #end loop
    
    
    #generating bigrams from TRAINING SET
    for tweet in tweets:
        bigrams = []
        #populating bigram list with bigrams
        bigrams = getBigramList(tweet[0])
        #adding the bigrams back to tweets List
        for bigram in bigrams:
            tweet[0].append(bigram)
    #end 
    
            
    #loop for building FeatureList ONLY FOR TRAINING DATA
    featureList = []    
    for tweet in tweets:    
        for word in tweet[0]:
            if(word in featureList):
                continue
            else:
                featureList.append(word.lower())
    #end of Loop for FeatureList Build
    
    
    #Getting sparse Matrix and getting Sentiment  for the Training Data in a seperate list in binary form    
    sparseMatrixTrain = getSparseMatrix(tweets, featureList)
    trainSentiment = getSentimentList(tweets)
    
    
    #Build tweets Array for Testing Data
    testTweets = [] 
    for line in rawTestData:
        sentiment = line[1]
        tweetText = line[0]
        processedTweet = processTweet(tweetText)
        featureVector = getFeatureVector(processedTweet, stopWords)
        testTweets.append((featureVector, sentiment));  
    #end loop
    
    #generating bigrams from the TEST SET
    for tweet in testTweets:
        bigrams = []
        #populating bigram list with bigrams
        bigrams = getBigramList(tweet[0])
        #adding the bigrams back to tweets List
        for bigram in bigrams:
            tweet[0].append(bigram)
    #end 
        
    #Getting sparse Matrix and getting Sentiment  for the Test Data in a seperate list in binary form    
    sparseMatrixTest = getSparseMatrix(testTweets, featureList)
    testSentiment = getSentimentList(testTweets)
    
    #Instantiating and training the SVM classifier
    SvmClassifier = svm.SVC(kernel ='linear')
    SvmClassifier.fit(sparseMatrixTrain, trainSentiment)
    
    #using classifier to predict Class for the Test data
    predictedTestSentiment = []
    predictedTestSentiment = SvmClassifier.predict(sparseMatrixTest) 
    
    acc = accuracy_score(testSentiment, predictedTestSentiment)
    print "\n Accuracy : "
    print acc 
    
    #return the Accuracy of the NBClassifier for the given Training and Test Data    
    return acc
 
#end of classifySciKit_SVM





print "Calling Main"    
if  __name__ =='__main__':main()    

