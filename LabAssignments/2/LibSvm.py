# Python2
import time 
import numpy as np
import pandas as pd
from svmutil import *



entry_no = 7
# start_time = time.time()
with open('./mnist/train.csv') as f:
	df = pd.read_csv(f,header=None)

df1 = (df.loc[(df.iloc[:,-1] == entry_no) | (df.iloc[:,-1] == (entry_no+1)%10)])

df1.iloc[:,-1]=df1.iloc[:,-1].replace({entry_no:1,(entry_no+1)%10:-1})

Y = np.array(df1.iloc[:,-1])
X = np.array(df1.iloc[:,:-1])
m = len(Y)
Y = Y.reshape(m,1)
C = 1.0
n = 28*28 #28 X 28 dimension images

Y_p = list(df1.iloc[:,-1])
X_p = df1.iloc[:,:-1].values.tolist()

# Linear Kernel
svm_model.predict = lambda self,y,x: svm_predict(y, x, self)[0][0]
prob = svm_problem(Y_p,X_p)
param = svm_parameter('-s 0 -t 0 -c '+str(C)) ## Linear SVM 
param.kernel_type = LINEAR 
param.C = C

model = svm_train(prob, param)

b = -model.rho.contents.value ## seen from documentation

training_accuracy = model.predict(Y_p,X_p)
support_vector_indices = model.get_sv_indices()
alphas = model.get_sv_coef() #corresponding to support vector indices other alphas are zeros
w = (X[np.array(support_vector_indices)-1].T).dot(Y[np.array(support_vector_indices)-1]*np.array(alphas).reshape(len(alphas),1))
if model.get_labels()[1] == -1:  # just to make vector point to positive examples
    w = -w
    b = -b
print("Training accuracy using Linear Kernel",training_accuracy*100)

# Gaussian Kernel
G=0.05 #as given in assignment
svm_model.predict = lambda self,y,x: svm_predict(y, x, self)[0][0]
prob = svm_problem(Y_p,X_p)
param = svm_parameter('-s 0 -t 2 -c '+str(C)+' -g '+str(G)) ## Gaussian SVM 


model = svm_train(prob, param)



training_accuracy = model.predict(Y_p,X_p)
print("Training Accuracy using Gassian Kerenl ",training_accuracy*100)

# Usefull link
# https://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f201
