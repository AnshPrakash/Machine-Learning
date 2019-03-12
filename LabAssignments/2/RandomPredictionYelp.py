# pyhton3
import pandas as pd
import numpy as np
from math import log
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from utils import *
import time 
import random
import sys



start_time=time.time()


test_loc=sys.argv[2]
training_loc=sys.argv[1]

with open(training_loc) as f:
	df = pd.read_csv(f)

reviews = list(df["text"])
stars = list(df["stars"]) #class 


N=len(df)
reviews =reviews[:N]
stars = list(map(int,stars[:N]))

pred_rand= [random.randint(1,5) for i in range(N)]
rand_conf=confusion_matrix(stars,pred_rand,labels=[1,2,3,4,5])
print("Confusion matrix by random prediction")
print(rand_conf)
print("accuracy by random prediction ",np.trace(rand_conf)/np.sum(rand_conf))
print("F1 score by random prediction policy")
f1=f1_score(stars, pred_rand, labels=[1,2,3,4,5],average=None)
print("F1 macro ",np.mean(f1))
print("F1 array ")
print(f1)
count=[0]*5
for star in stars:
	count[star-1]+=1

max_repeated_item=np.argmax(count)+1
# Max Prediction stategy
max_conf = confusion_matrix(stars,[max_repeated_item]*len(stars),labels=[1,2,3,4,5])
print("Confusion matrix by max repeated element policy")
print(max_conf)
print("Accuracy by max_prediction policy ",np.trace(max_conf)/np.sum(max_conf))
print("F1 score by max prediction policy")
f1=f1_score(stars,[max_repeated_item]*len(stars), labels=[1,2,3,4,5],average=None)
print("F1 macro ",np.mean(f1))
print("F1 array ")
print(f1)





