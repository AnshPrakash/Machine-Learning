# Python2 LibSVM Multi Class classification
import time 
import numpy as np
import pandas as pd
from svmutil import *

start_time=time.time()
with open('./mnist/train.csv') as f:
	df = pd.read_csv(f,header=None)

M=len(df)
df.iloc[:,:-1]=df.iloc[:,:-1]/255

Y_train = list(df.iloc[:,-1])
X_train = df.iloc[:,:-1].values.tolist()
dataframes=[] # list of data frame for each digit from 0-9
for i in xrange(10):
	dataframes.append(df.loc[(df.iloc[:,-1] == i)])



def training_models(C):
	global dataframes,M,X_train,Y_train
	G=0.05 # given in assignment Fixed
	svm_model.predict = lambda self,y,x: svm_predict(y, x, self)[0][0]
	# creatin 45 separate classification problems
	probs    = {}
	params   = {}
	models   = {}
	kc2_list = {}
	wins     = np.zeros((M,10),dtype=int)
	for i in xrange(9):#0-8
		for j in xrange(i+1,10): #every git grater than this one
			params[(i,j)]=(svm_parameter('-s 0 -t 2 -c '+str(C)+' -g '+str(G)))
			df_pos  = dataframes[i]
			df_neg  = dataframes[j]
			df_comb = df_pos.append(df_neg,ignore_index=True)
			df_comb.iloc[:,-1]=df_comb.iloc[:,-1].replace({i:1,j:-1})
			Y_p = list(df_comb.iloc[:,-1])
			X_p = df_comb.iloc[:,:-1].values.tolist()
			kc2_list[(i,j)]=(Y_p,X_p)
			probs[(i,j)]=(svm_problem(Y_p,X_p))
			models[(i,j)]=svm_train(probs[(i,j)],params[(i,j)])

	# Calculating training Set Accuracy
	for i in xrange(9):
		for j in xrange(i+1,10):
			results=svm_predict([0]*len(X_train),X_train,models[(i,j)])
			for k in xrange(len(results[0])):
				if results[0][k]==1:
					wins[k][i] +=1
				else:
					wins[k][j] +=1
	preds=len(wins[0]) - np.argmax(np.flip(wins,1),axis=1)-1
	print("Training set Accuracy :",np.mean(preds==Y_train))	
	return(models)




models=training_models(1.0)
end_time=time.time()
print("Time Taken ",end_time- start_time)