#Classifies email snippets as spam or non-spam, using a Naive Bayes' Classifier
#Spam is defined as any snippet where P(Spam) > P(Non-Spam)
import pandas as pd

def classifier():
    print("in classifier")

def trainer():
    trainingSet = pd.read_csv("dataset.csv")
    print(trainingSet)
    msgCount = 0
    numSpam = 0
    numHam = 0

    print(trainingSet['v1'])

    for row in trainingSet.iterrows():
        answer = row[1]['v1']
        msg = row[1]['v2']
        print(msg)


        if (answer == "spam"):
            numSpam += 1
        else:
            numHam += 1
        msgCount += 1

    print("NumSpam: " + str(numSpam))
    print("NumHam: " + str(numHam))
    print("Total Messages: " + str(msgCount))


def tester():
    myVar = 0
    #print("hi")


if __name__ == "__main__":
    classifier()
    trainer()
    tester()