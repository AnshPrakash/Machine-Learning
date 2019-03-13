# Python2 LibSVM Multi Class classification
import time 
import numpy as np
import pandas as pd
import sys
sys.path.append("./libsvm-3.23/python")
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
	param  = svm_parameter('-s 0 -t 2 -c '+ str(C)+' -g '+str(G))
	prob   = svm_problem(Y_train,X_train)
	model  = svm_train(prob,param)
	result = svm_predict(Y_train,X_train,model)
	print("Training set Accuracy :",result[1][0])	
	return(model)

model=training_models(1.0)

with open('./mnist/test.csv') as f:
	df = pd.read_csv(f,header=None)

df.iloc[:,:-1]=df.iloc[:,:-1]/255
Y_test = list(df.iloc[:,-1])
X_test = df.iloc[:,:-1].values.tolist()
print("Test Set Prediction Accuracy: ",svm_predict(Y_test,X_test,model)[1][0])


end_time=time.clock()
print("Time Taken ",end_time- start_time)