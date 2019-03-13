# Assignment2 part2 2 (d) python3
import time 
import numpy as np
import pandas as pd
import sys
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
sys.path.append("./libsvm-3.23/python")
from svmutil import *


test_f =sys.argv[2]
train_f=sys.argv[1]

start_time=time.clock()
with open(train_f) as f:
	df = pd.read_csv(f,header=None)

M=len(df)
df = shuffle(df)
df.iloc[:,:-1]=df.iloc[:,:-1]/255

Y_train = list(df.iloc[:,-1])
X_train = df.iloc[:,:-1].values.tolist()


Y_train,Y_valid =Y_train[:int(M*0.9)],Y_train[int(M*0.9):]
X_train,X_valid =X_train[:int(M*0.9)],X_train[int(M*0.9):]



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
	param  = svm_parameter('-s 0 -t 2 -c '+ str(C)+' -g '+str(G)+' -h 0')
	prob   = svm_problem(Y_train,X_train)
	model  = svm_train(prob,param)
	return(model)

C_pos=[1e-5,1e-3,1.0,5.0,10]
models={}
for C in C_pos:
	models[C]=training_models(C)

acc_val={}
val_acc=[]
for C in C_pos:
	acc_val[C]=svm_predict(Y_valid,X_valid,models[C])[1][0]
	val_acc.append(acc_val[C])



with open(test_f) as f:
	df = pd.read_csv(f,header=None)

df.iloc[:,:-1]=df.iloc[:,:-1]/255
Y_test = list(df.iloc[:,-1])
X_test = df.iloc[:,:-1].values.tolist()
test_acc ={}
acc_test =[]
for C in C_pos:
	test_acc[C] = svm_predict(Y_test,X_test,models[C])[1][0]
	acc_test.append(test_acc[C])

print("Validation set accuracy")
print(acc_val)
print("Test set accuracy")
print(test_acc)
plt.plot(acc_test,val_acc)
plt.ylabel('Validation Set Accuracy')
plt.xlabel('Test Set Accuacy')
plt.show()

end_time=time.clock()
print("Time Taken ",end_time- start_time)