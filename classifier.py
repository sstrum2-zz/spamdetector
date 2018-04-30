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

trainPositive = {}
trainNegative = {}
numSpam = 0
numHam = 0
msgCount = 0

def trainer():
    global numSpam
    global numHam
    global msgCount

    trainingSet = pd.read_csv("dataset_train.csv")

    spamWords = ""
    hamWords = ""

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
              #  print(word)
                trainPositive[word] = trainPositive.get(word, 0) + 1
            numSpam = numSpam + 1

        else:
            for word in msg:
                hamWords += " "
                hamWords += word
            #    print(word)
                trainNegative[word] = trainNegative.get(word, 0) + 1
            numHam = numHam + 1

        msgCount += 1


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

def bayesProbability_tokens(token, isSpam):
    p = 0
    smoothing_factor = .7
    if isSpam:
        num = trainPositive.get(token,0)+ smoothing_factor
        den = float(numSpam + (smoothing_factor * (numSpam + numHam)))
        p = num/den

        return p
    else:
        num = trainNegative.get(token, 0) + smoothing_factor
        den = float(numHam + (smoothing_factor * (numSpam + numHam)))
        p = num/den

        return p

#multiply all token probabilities together for each message
def bayesProb_msg(msg, isSpam):
    p = 1.0
    for word in msg.split(" "):
        word = word.lower()
        p = p * bayesProbability_tokens(word, isSpam)
    return p

def classify_msg(msg):
    global numSpam
    global numHam
    #decides whether or not an individual message in the test set is spam
    pSpamOverall = 100 * float(numSpam)/float(numSpam + numHam)
 #   print(numSpam)
 #   print(numSpam + numHam)
    pHamOverall = 100 * float(numHam)/float((numSpam + numHam))
 #   print(pSpamOverall)
 #   print(bayesProb_msg(msg, True))
 #   print(bayesProb_msg(msg, False))

    isSpam = pSpamOverall * bayesProb_msg(msg, True)
    notSpam = pHamOverall * bayesProb_msg(msg, False)


   # print("P(SPAM):" + str(isSpam))
  #  print("P(HAM):" + str(notSpam))

    if (isSpam > notSpam):
       # print("LIKELY SPAM")
        return True
    else:
       # print("LIKELY HAM")
        return False


if __name__ == "__main__":
    numAccurate = 0
    numWrong = 0
    truePos = 0
    falsePos = 0
    falseNeg = 0
    trueNeg = 0
    msgCount1 = 0

    trainer()
    testingSet = pd.read_csv("dataset_test.csv")

    for row in testingSet.iterrows():
        answer = row[1]['v1']
        msg = row[1]['v2']
        detectedSpam = classify_msg(msg)
        print(detectedSpam)

        if (detectedSpam) and (answer == "spam"):
            numAccurate = numAccurate + 1
            truePos = truePos + 1
        elif (detectedSpam) and (answer == "ham"):
            falsePos = falsePos + 1
            numWrong = numWrong + 1
        elif (detectedSpam == False) and (answer == "ham"):
            trueNeg = trueNeg + 1
            numAccurate = numAccurate + 1
        else:
            falseNeg = falseNeg + 1
            numWrong = numWrong + 1

        msgCount1 += 1

    print("Percent True Positives: " + str((float(truePos)/float(msgCount1))))
    print("Percent True Negatives: " + str((float(trueNeg)/float(msgCount1))))
    print("Percent False Positives: " + str((float(falsePos)/float(msgCount1))))
    print("Percent False Negatives: " + str((float(falseNeg)/float(msgCount1))))
    print ("Accuracy Rate: " + str(float(numAccurate)/float(msgCount1)))