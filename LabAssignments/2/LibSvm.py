# Python3
import time 
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import sys
sys.path.append("./libsvm-3.23/python")
from svmutil import *

test_f =sys.argv[2]
train_f=sys.argv[1]

entry_no = 7
# start_time = time.time()
with open(train_f) as f:
	df = pd.read_csv(f,header=None)

df1 = (df.loc[(df.iloc[:,-1] == entry_no) | (df.iloc[:,-1] == (entry_no+1)%10)])

df1.iloc[:,-1]=df1.iloc[:,-1].replace({entry_no:1,(entry_no+1)%10:-1})
df1.iloc[:,:-1]=df1.iloc[:,:-1]/255

Y = np.array(df1.iloc[:,-1])
X = np.array(df1.iloc[:,:-1])
m = len(Y)
Y = Y.reshape(m,1)
C = 1.0
n = 28*28 #28 X 28 dimension images

Y_p = list(df1.iloc[:,-1])
X_p = df1.iloc[:,:-1].values.tolist()

# Linear Kernel
# svm_model.predict = lambda self,y,x: svm_predict(y, x, self)[0][0]
prob = svm_problem(Y_p,X_p)
param = svm_parameter('-s 0 -t 0 -c '+str(C)) ## Linear SVM 
param.kernel_type = LINEAR 
param.C = C

model = svm_train(prob, param)

b = -model.rho.contents.value ## seen from documentation

res = svm_predict(Y_p,X_p,model)
training_accuracy = res[1][0]
support_vector_indices = model.get_sv_indices()
alphas = model.get_sv_coef() #corresponding to support vector indices other alphas are zeros
w = (X[np.array(support_vector_indices)-1].T).dot(Y[np.array(support_vector_indices)-1]*np.array(alphas).reshape(len(alphas),1))
if model.get_labels()[1] == -1:  # just to make vector point to positive examples
    w = -w
    b = -b
print("Weight vector w for Linear Kernel: ")
# print(w)
print("bias term for Linear Kernel")
print(b)
print("Training accuracy using Linear Kernel",training_accuracy)
print("Confusion matrix for Linear Kernel ")
print(confusion_matrix(Y_p,res[0],labels=[-1,1]))
# Gaussian Kernel
G=0.05 #as given in assignment
# svm_model.predict = lambda self,y,x: svm_predict(y, x, self)[0][0]
prob = svm_problem(Y_p,X_p)
param = svm_parameter('-s 0 -t 2 -c '+str(C)+' -g '+str(G)) ## Gaussian SVM 


model_g = svm_train(prob, param)

b = -model_g.rho.contents.value ## seen from documentation
if model_g.get_labels()[1] == -1:  # just to make vector point to positive examples
    b = -b
print("Bias term for gaussian kernel is ",b)
print("weight vector for gaussian kernel is infinity dimension vector,So it cannot be explicitly calculated")

res_g = svm_predict(Y_p,X_p,model_g)

training_accuracy = res_g[1][0]
print("Training Accuracy using Gassian Kerenl ",training_accuracy)
print("Confusion MAtrix for Gaussian Kernel")
print(confusion_matrix(Y_p,res_g[0],labels=[-1,1]))




# traing Set accuracy

with open(test_f) as f:
	df = pd.read_csv(f,header=None)

df1 = (df.loc[(df.iloc[:,-1] == entry_no) | (df.iloc[:,-1] == (entry_no+1)%10)])

df1.iloc[:,-1]=df1.iloc[:,-1].replace({entry_no:1,(entry_no+1)%10:-1})
df1.iloc[:,:-1]=df1.iloc[:,:-1]/255


Y_test = list(df1.iloc[:,-1])
X_test = df1.iloc[:,:-1].values.tolist()


res = svm_predict(Y_test,X_test,model)
accuracy = res[1][0]
print("Test Set accuracy using Linear Kernel",accuracy)
print("Confusion matrix for Linear Kernel ")
print(confusion_matrix(Y_test,res[0],labels=[-1,1]))
print("\n")



res_g = svm_predict(Y_test,X_test,model_g)

accuracy = res_g[1][0]
print("Test Accuracy using Gassian Kerenl ",accuracy)
print("Confusion Matrix for Gaussian Kernel")
print(confusion_matrix(Y_test,res_g[0],labels=[-1,1]))


# Usefull link
# https://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f201
