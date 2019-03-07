# pyhton3
import pandas as pd
import numpy as np
from math import log
from sklearn.metrics import confusion_matrix
import time 


start_time=time.time()
with open('./ass2_data/train.csv') as f:
	df = pd.read_csv(f)

reviews = list(df["text"])
stars = list(df["stars"]) #class 

# Consider this many traning points
N =len(df) 
reviews = reviews[:N]
# stars=stars[:N]
stars = list(map(int,stars[:N]))


##Making Vocablary
vocab = {}
for review in reviews:
	text = review.split()
	for word in text:
		if word not in vocab:
			vocab[word] = [0.0,0.0,0.0,0.0,0.0]

V=len(vocab)

p_stars = [0.0,0.0,0.0,0.0,0.0]
doc_words = [0,0,0,0,0] ## total number of word in each class
for i in range(len(reviews)):
	text = reviews[i].split()
	doc_words[stars[i]-1] += len(text)
	p_stars[stars[i]-1] += 1.0
	for word in text:
		vocab[word][stars[i]-1] += 1.0

print(p_stars)
p_stars = list(map(lambda x:x/N,p_stars))
print(p_stars)




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
		text=test_reviews[i].split()
		p=list(map(log,p_stars))
		for j in range(5):
			for word in text:
				p[j]+=log((1+vocab[word][j])/(V+doc_words[j]))
		pred_star=np.argmax(p)+1
		pred[i] = pred_star
		corr=corr+1 if label_stars[i]==pred_star else corr
	# get the confusion matrix
	c=confusion_matrix(label_stars,pred,labels=[1,2,3,4,5])
	print(c)
	return(corr/M)

print("Training Set accuracy",getaccuracy(stars,reviews))

# Load test data
with open('./ass2_data/test.csv') as f:
	df2 = pd.read_csv(f)

test_reviews = list(df2["text"])
test_stars = list(df2["stars"]) #class 

# Consider this many traning points
N_test =len(df2) 
test_reviews = reviews[:N_test]
# stars=stars[:N]
test_stars = list(map(int,stars[:N_test]))
print("Test Set accuracy",getaccuracy(test_stars,test_reviews))
end_time=time.time()
print("Time taken by the code is ",end_time - start_time)