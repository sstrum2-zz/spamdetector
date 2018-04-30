
Detecting Spam Text Messages Using a Naive Bayes' Classifier
============


Introduction
---------

As spam text messages are becoming more sophisticated and difficult to detect by humans,
we can apply analytical techniques to better predict the presence of spamming attempts. This report seeks to create 
a model to efficiently detect spam text messages and provide an analysis of the most common phrases that could
signify a spam attempt. 

Methodology
----------

The o

Analysis of Results
-----

-   Understand the basic syntax of Scheme and how to evaluate programs in it
-   Understand how to simulate stateful computation by composing monads, and
    


Relevant Files
--------------

In the directory `app/`, you will find the program code, some of which is only
partially implemented, and which you will have to modify to complete this
assignment. The files `test/Spec.hs` and `test/Tests.hs` contain the code used for testing.

Running the Classifier
------------

To run this code, navigate to the spamdetector directory and execute the following command:

``` {.sh}
$ python classifier.py
```
Note that you may have to install certain dependencies (e.g. the "punkt" package for punctuation detection). For the given files (dataset.csv), the following results will be printed to the terminal: 
``` {.sh}
Percent True Positives: 0.130227001195
Percent True Negatives: 0.745519713262
Percent False Positives: 0.118279569892
Percent False Negatives: 0.00597371565114
Accuracy Rate: 0.875746714456

```


