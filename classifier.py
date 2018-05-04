#Classifies text message snippets as spam or non-spam, using a Naive Bayes' Classifier
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
foundspamwords = ""

def trainer():
    global numSpam
    global numHam
    global msgCount

    trainingSet = pd.read_csv("dataset_train.csv")

    spamWords = ""
    hamWords = ""

    for row in trainingSet.iterrows():
        answer = row[1]['v1']
        msg = row[1]['v2']

        msg = porter_stemmer(msg)

        if (answer == "spam"):
            for word in msg:
                spamWords += " "
                spamWords += word
                trainPositive[word] = trainPositive.get(word, 0) + 1
            numSpam = numSpam + 1

        else:
            for word in msg:
                hamWords += " "
                hamWords += word
                trainNegative[word] = trainNegative.get(word, 0) + 1
            numHam = numHam + 1

        msgCount += 1

    print("\n")
    print("Training Spam WordCloud opening in new window...")
    print("\n")
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

    #remove stop words
    stop_word = stopwords.words('english')
    tokens = [t for t in tokens if (t not in stop_word)]

    #lastly, run the porter stemmer algorithm on the words
    stemfxn = PorterStemmer()
    tokens = [stemfxn.stem(t) for t in tokens]

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
    global foundspamwords

    #decides whether or not an individual message in the test set is spam
    pSpamOverall = 100 * float(numSpam)/float(numSpam + numHam)
    pHamOverall = 100 * float(numHam)/float((numSpam + numHam))

    isSpam = pSpamOverall * bayesProb_msg(msg, True)
    notSpam = pHamOverall * bayesProb_msg(msg, False)


    if (isSpam > notSpam):
       # likely spam
        foundspamwords += " "
        foundspamwords += msg
        return True
    else:
       # likely not spam
        return False


if __name__ == "__main__":
    numAccurate = 0
    numWrong = 0
    truePos = 0
    falsePos = 0
    falseNeg = 0
    trueNeg = 0
    totalMessageCount = 0

    trainer()
    testingSet = pd.read_csv("dataset_test.csv")

    for row in testingSet.iterrows():
        answer = row[1]['v1']
        msg = row[1]['v2']
        detectedSpam = classify_msg(msg)

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

        totalMessageCount += 1

    print("Testing Spam WordCloud opening in new window...")
    print("\n")
    myspamwc = WordCloud(width=512, height=512).generate(foundspamwords)
    plt.figure(figsize=(10, 8), facecolor='w')
    plt.imshow(myspamwc)
    plt.title("Common Spam Words From Testing Dataset")
    plt.show()

    print("True Positives: " + str((float(truePos) / float(totalMessageCount))))
    print("True Negatives: " + str((float(trueNeg) / float(totalMessageCount))))
    print("False Positives: " + str((float(falsePos) / float(totalMessageCount))))
    print("False Negatives: " + str((float(falseNeg) / float(totalMessageCount))))
    print ("Accuracy Rate: " + str(float(numAccurate) / float(totalMessageCount)))