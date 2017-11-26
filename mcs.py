import os
import numpy as np
from random import randint as rr
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

def make_Corpus(root_dir):
    polarity_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]
    corpus = []
    for polarity_dir in polarity_dirs:
        reviews = [os.path.join(polarity_dir,f) for f in os.listdir(polarity_dir)]
        for review in reviews:
            doc_string = "";
            with open(review) as rev:
                for line in rev:
                    doc_string = doc_string + line
            if not corpus:
                corpus = [doc_string]
            else:
                corpus.append(doc_string)
    return corpus

#Create a corpus with each document having one string
root_dir = 'train'
train_corpus = make_Corpus(root_dir)
root_dir = 'test'
test_corpus = make_Corpus(root_dir)

#create labels
labels = np.zeros(25000);
labels[0:12500]=0;
labels[12500:25000]=1;

vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True,stop_words='english')
train_corpus_tf_idf = vectorizer.fit_transform(train_corpus)
test_corpus_tf_idf = vectorizer.transform(test_corpus)

vocabulary = vectorizer.get_feature_names()
size_of_vocabulary = len(vocabulary)

positive_vector = np.zeros(len(vocabulary))
negative_vector = np.zeros(len(vocabulary))
negative_tf_idf_sum = 0
positive_tf_idf_sum = 0

for i in range(12500):
    negative_vector = np.add(negative_vector,train_corpus_tf_idf[i].toarray())

for i in range(size_of_vocabulary):
    negative_tf_idf_sum = negative_tf_idf_sum + negative_vector[0][i]

for i in range(12501,25000):
    positive_vector = np.add(positive_vector,train_corpus_tf_idf[i].toarray())

for i in range(size_of_vocabulary):
    positive_tf_idf_sum = positive_tf_idf_sum + 1

total_tf_idf = negative_tf_idf_sum+ positive_tf_idf_sum

negative_probability = [x/(negative_tf_idf_sum+total_tf_idf) for x in negative_vector]
positive_probability = [x/(positive_tf_idf_sum+total_tf_idf) for x in positive_vector]

correct = 0
incorrect = 0
confusing = 0

print_correct = 5
print_incorrect = 5

correct_vector = []
incorrect_vector = []

print "Starting Checking......"

for i in range(25000):
    prediction = 0.0
    result = 0
    test_data = test_corpus_tf_idf[i].toarray()
    for j in range(size_of_vocabulary):
        if(positive_probability[0][j]>negative_probability[0][j]):
            prediction = prediction + (test_data[0][j] * positive_probability[0][j])
        elif(positive_probability[0][j]<negative_probability[0][j]):
            prediction = prediction - (test_data[0][j] * negative_probability[0][j])

    if(prediction>0.0):
        result = 1
    elif(prediction<0.0):
        result = 0
    else:
        result = -99.0

    if(result==labels[i]):
        correct = correct + 1
        correct_vector.append(prediction)
        if(rr(1,10)<3 and print_correct > 0):
            print 'Prediction:'+str(prediction)+'\nCorrect!!!!\nReview:\n'+test_corpus[i]+'\n\n'
            print_correct = print_correct - 1
    elif(result== -99.0):
        confusing = confusing + 1
    else:
        incorrect = incorrect + 1
        incorrect_vector.append(prediction)
        if(rr(1,10)<3 and print_incorrect > 0):
            print 'Prediction:'+str(prediction)+'\nInorrect!!!!\nReview:\n'+test_corpus[i]+'\n\n'
            print_incorrect = print_incorrect - 1

print 'Correct:'+str(correct)+'\nIncorrect:'+str(incorrect)+'\nConnfusing:'+str(confusing)+'\n\n'
print 'RMS of correct_vector:' + str(np.sqrt(np.mean((np.array(correct_vector)**2))))+'\n\n'
print 'RMS of incorrect_vector:' + str(np.sqrt(np.mean((np.array(incorrect_vector)**2))))+'\n\n'
