# pyhton3 BIGRAMS
import pandas as pd
import numpy as np
from math import log
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from utils import *
import time 
import sys
import nltk

def getbigrams(doc):
	# print("before preprocessing ",len(doc))
	doc=re.sub(r'[^\w\s]','',doc.lower())
	text=doc.split()
	gen=list(nltk.bigrams(text))
	txt=[]
	for word in gen:
		txt.append(word[0]+" "+word[1])
	# print("after preprocessing ",len(txt))
	return(txt)




start_time=time.time()

# part_num=sys.argv[3]
test_loc=sys.argv[2]
training_loc=sys.argv[1]
# training_loc="./ass2_data/train.csv"
# test_loc="./ass2_data/test.csv"

with open(training_loc) as f:
	df = pd.read_csv(f)

reviews = list(df["text"])
stars = list(df["stars"]) #class 

N=len(df)
reviews =reviews[:N]
stars = list(map(int,stars[:N]))

time_taken=time.time()
print("Time Taken for loading data",time_taken- start_time)
print("Bigramming Started Start")
new_reviews =[getbigrams(review) for review in reviews ]
reviews=new_reviews
print("Bigramming Done")
time_taken2=time.time()
print("Time taken for Bigraming ", time_taken2- time_taken)
vocab={}
for review in reviews:
	for word in review:
		if word not in vocab:
			vocab[word] = [0.0,0.0,0.0,0.0,0.0]
V=len(vocab)
p_stars = [0.0,0.0,0.0,0.0,0.0]
doc_words = [0,0,0,0,0] ## total number of word in each class

for i in range(len(reviews)):
	text = reviews[i]
	doc_words[stars[i]-1] += len(text)
	p_stars[stars[i]-1] += 1.0
	for word in text:
		vocab[word][stars[i]-1] += 1.0

print(p_stars)
p_stars = list(map(lambda x:x/N,p_stars))
print(p_stars)

for word in vocab:
	for j in range(5):
		vocab[word][j]=log((1+vocab[word][j])/(V+doc_words[j]))

def getaccuracy(label_stars,test_reviews):
	# get accuracy of trainig set
	global vocab
	global doc_words
	global V
	global p_stars
	M=len(label_stars)
	pred=[0]*M
	corr=0
	for i in range(len(test_reviews)):
		text=test_reviews[i]
		p=list(map(log,p_stars))
		for j in range(5):
			for word in text:
				try:
					p[j]+=vocab[word][j]
				except Exception as e:
					p[j]+=log(1/(V+doc_words[j]))
		pred_star=np.argmax(p)+1
		pred[i] = pred_star
		corr=corr+1 if label_stars[i]==pred_star else corr
	# get the confusion matrix
	c=confusion_matrix(label_stars,pred,labels=[1,2,3,4,5])
	print(c)
	return(corr/M)

print("Training Set accuracy",getaccuracy(stars,reviews))

# Load test data
with open(test_loc) as f:
	df2 = pd.read_csv(f)

test_reviews = list(df2["text"])
test_stars = list(df2["stars"]) #class 

# Consider this many test points
N_test = len(df2) 
test_reviews = test_reviews[:N_test]
# stars=stars[:N]
test_stars = list(map(int,test_stars[:N_test]))
test_reviews =[getbigrams(review) for review in test_reviews ]


print("Test Set accuracy",getaccuracy(test_stars,test_reviews))

