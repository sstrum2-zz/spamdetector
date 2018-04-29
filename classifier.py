#Classifies email snippets as spam or non-spam, using a Naive Bayes' Classifier
#Spam is defined as any snippet where P(Spam) > P(Non-Spam)
import pandas as pd

import matplotlib.pyplot as plt
from wordcloud import WordCloud

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

def trainer():
    trainPositive = {}
    trainingSet = pd.read_csv("dataset_train.csv")
  #  print(trainingSet)
    msgCount = 0
    numSpam = 0
    numHam = 0
    spamWords = ""
    ham_msgs = []

  #  print(trainingSet['v1'])

    for row in trainingSet.iterrows():
        answer = row[1]['v1']
        msg = row[1]['v2']
       # print(msg)
        msg = porter_stemmer(msg)


        if (answer == "spam"):
            for word in msg:
                spamWords += " "
                spamWords += word
                print(word)
                trainPositive[word] = trainPositive.get(word, 0) + 1
            numSpam = numSpam + 1

        else:
            ham_msgs.append(msg)
            numHam += 1
        msgCount += 1

  #  print("NumSpam: " + str(numSpam))
  #  print("NumHam: " + str(numHam))
  #  print("Total Messages: " + str(msgCount))
    print(trainPositive)

    spamwc = WordCloud(width=512, height=512).generate(spamWords)
    plt.figure(figsize=(10,8), facecolor='w')
    plt.imshow(spamwc)
    plt.title("Common Spam Words From Training Dataset")
    plt.show()

def porter_stemmer(message):

    #transform message into tokens for analysis
    tokens = word_tokenize(message.decode('latin-1'))

    #remove words of short length (< 2)
    tokens = [t for t in tokens if len(t) > 2]

    #insert n-grams here for accuracy enhancement

    #remove stop words
    stop_word = stopwords.words('english')
    tokens = [t for t in tokens if (t not in stop_word)]

    #lastly, run the porter stemmer algorithm on the words
    stemfxn = PorterStemmer()
    tokens = [stemfxn.stem(t) for t in tokens]

   # print(tokens)
    return tokens

def tester():
    myVar = 0
    #print("hi")


if __name__ == "__main__":
    trainer()
    tester()