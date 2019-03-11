# pyhton3
import pandas as pd
import numpy as np
from math import log
from sklearn.metrics import confusion_matrix
from utils import *
import time 
import sys




start_time=time.time()

part_num=sys.argv[3]
test_loc=sys.argv[2]
training_loc=sys.argv[1]
if part_num=="a" or part_num =="c" or part_num=="d":

	with open(training_loc) as f:
		df = pd.read_csv(f)

	reviews = list(df["text"])
	stars = list(df["stars"]) #class 

	# Consider this many traning points
	N = len(df) 
	reviews = reviews[:N]
	# stars=stars[:N]
	stars = list(map(int,stars[:N]))

	# reviews=map(getStemmedDocuments,reviews)
	##Making Vocablary
	def getvocab(part):
		vocab = {}
		if part=="d":
			for review in reviews:

				text = getStemmedDocuments(review)
				for word in text:
					if word not in vocab:
						vocab[word] = [0.0,0.0,0.0,0.0,0.0]
		else:
			for review in reviews:
				text = re.sub(r'[^\w\s]','',review.lower())
				text = text.split()
				# text = review.split()
				for word in text:
					if word not in vocab:
						vocab[word] = [0.0,0.0,0.0,0.0,0.0]
		return(vocab)

	vocab=getvocab(part_num)
	V=len(vocab)

	p_stars = [0.0,0.0,0.0,0.0,0.0]
	doc_words = [0,0,0,0,0] ## total number of word in each class
	if part_num=="d":
		for i in range(len(reviews)):
			text = getStemmedDocuments(reviews[i])
			doc_words[stars[i]-1] += len(text)
			p_stars[stars[i]-1] += 1.0
			for word in text:
				vocab[word][stars[i]-1] += 1.0
	else:	
		for i in range(len(reviews)):
			text = re.sub(r'[^\w\s]','',reviews[i].lower())
			text = text.split()
			# text = reviews[i].split()
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
			text = re.sub(r'[^\w\s]','',test_reviews[i].lower())
			text=text.split()
			# text=test_reviews[i].split()
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

	def getStop_stemmed_accuracy(label_stars,test_reviews):
		# get accuracy of trainig set
		global vocab
		global doc_words
		global V
		global p_stars
		M=len(label_stars)
		pred=[0]*M
		corr=0
		for i in range(len(test_reviews)):
			text=getStemmedDocuments(test_reviews[i])
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


	if part_num=="a":
		print("Training Set accuracy",getaccuracy(stars,reviews))
	if part_num=="d":
		print("Training set accuracy using stemming",getStop_stemmed_accuracy(stars,reviews))


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

	if part_num=="d":
		print("Test Set accuracy with stemmed and stopword preprocessing",getStop_stemmed_accuracy(test_stars,test_reviews))
	else:
		print("Test Set accuracy",getaccuracy(test_stars,test_reviews))


end_time=time.time()
print("Time taken by the code is ",end_time - start_time)

