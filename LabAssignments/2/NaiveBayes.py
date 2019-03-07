import pandas as pd
import numpy as np
from math import log

with open('./ass2_data/train.csv') as f:
	df = pd.read_csv(f)

reviews = list(df["text"])
stars = list(df["stars"]) #class 

# Consider this many traning points
N = len(df) #100
reviews = reviews[:N]
# stars=stars[:N]
stars = list(map(int,stars[:N]))


# print(reviews)
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

# get accuracy of trainig set



