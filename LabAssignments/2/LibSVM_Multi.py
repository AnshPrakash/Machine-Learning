# Python2 LibSVM Multi Class classification
import time 
import numpy as np
import pandas as pd
from svmutil import *

start_time=time.clock()
with open('./mnist/train.csv') as f:
	df = pd.read_csv(f,header=None)

M=len(df)
df.iloc[:,:-1]=df.iloc[:,:-1]/255

Y_train = list(df.iloc[:,-1])
X_train = df.iloc[:,:-1].values.tolist()


def get_training_data(i,j):
	global df
	df1=(df.loc[(df.iloc[:,-1]==i) | (df.iloc[:,-1]==j)])
	df1.iloc[:,-1]=df1.iloc[:,-1].replace({i:1,j:-1})
	Y_p = list(df1.iloc[:,-1])
	X_p = df1.iloc[:,:-1].values.tolist()
	return(Y_p,X_p)



def training_models(C):
	global M,X_train,Y_train
	G=0.05 # given in assignment Fixed
	svm_model.predict = lambda self,y,x: svm_predict(y, x, self)[0][0]
	# creatin 45 separate classification problems
	probs    = {}
	params   = {}
	models   = {}
	for i in xrange(9):#0-8
		for j in xrange(i+1,10): #every git grater than this one
			params[(i,j)]=(svm_parameter('-s 0 -t 2 -c '+str(C)+' -g '+str(G)))
			Y_p,X_p = get_training_data(i,j)
			probs[(i,j)] = svm_problem(Y_p,X_p)
			models[(i,j)]= svm_train(probs[(i,j)],params[(i,j)])

	# Calculating training Set Accuracy
	wins     = np.zeros((M,10),dtype=int)
	for i in xrange(9):
		for j in xrange(i+1,10):
			results=svm_predict(Y_train,X_train,models[(i,j)])
			for k in xrange(len(results[0])):
				if results[0][k]==1:
					wins[k][i] +=1
				else:
					wins[k][j] +=1
	preds=len(wins[0]) - np.argmax(np.flip(wins,1),axis=1)-1
	print("Training set Accuracy :",np.mean(preds==Y_train))	
	return(models)




models=training_models(1.0)
end_time=time.clock()
print("Time Taken ",end_time- start_time)