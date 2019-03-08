import time 
import numpy as np
import pandas as pd
from cvxopt import matrix
from cvxopt import solvers

entry_no=7
start_time=time.time()
with open('./mnist/train.csv') as f:
	df = pd.read_csv(f,header=None)

df1=(df.loc[(df.iloc[:,-1]==entry_no) | (df.iloc[:,-1]==(entry_no+1)%10)])


Y = np.array(df1.iloc[:,-1])
X = np.array(df1.iloc[:,:-1])/255
m=len(Y)
C = 1.0
Y=Y.reshape(m,1)
n=28*28

print(m)

# def linearKernelSVM():
# 	global X,Y,m,n,C
P = matrix((Y@(Y.T))*(X@(X.T)),tc='d')
q = matrix(-1*np.ones((m,1)),tc='d')
G = matrix(np.vstack((-np.identity(m),np.identity(m))),tc='d')
A = matrix(Y.T,tc='d')
h = matrix(np.vstack((np.zeros((m,1)),C*np.ones((m,1)))),tc='d')
b = matrix(np.zeros((1,1)))

sol=solvers.qp(P,q,G,h,A,b)
alphas=np.array(sol['x'])

w=(X.T)@alphas



# linearKernelSVM()