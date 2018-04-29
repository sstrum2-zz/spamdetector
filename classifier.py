#Classifies email snippets as spam or non-spam, using a Naive Bayes' Classifier
#Spam is defined as any snippet where P(Spam) > P(Non-Spam)
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def classifier():
    print("in classifier")

def trainer():
    trainingSet = pd.read_csv("dataset_train.csv")
    print(trainingSet)
    msgCount = 0
    numSpam = 0
    numHam = 0
    spamWords = ""
    ham_msgs = []

    print(trainingSet['v1'])

    for row in trainingSet.iterrows():
        answer = row[1]['v1']
        msg = row[1]['v2']
        print(msg)

        if (answer == "spam"):
            for word in msg.split():
                spamWords += " "
                spamWords += word
                print(word)
            numSpam += 1
        else:
            ham_msgs.append(msg)
            numHam += 1
        msgCount += 1

    print("NumSpam: " + str(numSpam))
    print("NumHam: " + str(numHam))
    print("Total Messages: " + str(msgCount))

    spamwc = WordCloud(width=512, height=512).generate(spamWords)
    plt.figure(figsize=(10,8), facecolor='k')
    plt.imshow(spamwc)
    plt.show()

def tester():
    myVar = 0
    #print("hi")


if __name__ == "__main__":
    classifier()
    trainer()
    tester()