# Python 3
import time 
import numpy as np
import pandas as pd
from cvxopt import matrix
from cvxopt import solvers
from sklearn.metrics import confusion_matrix

entry_no=7
start_time=time.time()
with open('./mnist/train.csv') as f:
	df = pd.read_csv(f,header=None)

df1=(df.loc[(df.iloc[:,-1]==entry_no) | (df.iloc[:,-1]==(entry_no+1)%10)])

df1.iloc[:,-1]=df1.iloc[:,-1].replace({entry_no:1,(entry_no+1)%10:-1})
df.iloc[:,:-1]=df.iloc[:,:-1]/255
Y = np.array(df1.iloc[:,-1])
X = np.array(df1.iloc[:,:-1])
m=len(Y)
C = 1.0
Y=Y.reshape(m,1)
n=28*28 #28 X 28 dimension images

print(m)

def linearKernelSVM():
	# returns the P in CVXOPT standard equation
	global X,Y
	return(matrix((Y@(Y.T))*(X@(X.T)),tc='d'))

gamma=0.05
def GaussianKernel():
	global X,Y
	Z=X@(X.T)
	P=np.zeros((m,m))
	for i in range(m):
		for j in range(m):
			P[i][j]=Z[i][i]+Z[j][j]-2*Z[i][j]
	P=matrix(Y@(Y.T)*np.exp(-gamma*P),tc='d')
	return(P)

P = linearKernelSVM()
q = matrix(-1*np.ones((m,1)),tc='d')
G = matrix(np.vstack((-np.identity(m),np.identity(m))),tc='d')
A = matrix(Y.T,tc='d')
h = matrix(np.vstack((np.zeros((m,1)),C*np.ones((m,1)))),tc='d')
b = matrix(np.zeros((1,1)))
sol=solvers.qp(P,q,G,h,A,b)
alphas=np.array(sol['x'])


# get the support vector
def getSupportVectorIndices(alphas,e=1e-4):
     vec=[]
     for i in range(m):
             if abs(alphas[i])>e:
                     vec.append(i)
     return(vec)

sv=getSupportVectorIndices(alphas,1e-4)

w = (X.T).dot(alphas*Y)
bias = 0.0
for i in sv:
	if alphas[i]<C:
		bias = Y[i] - np.dot(w.T,X[i].reshape(n,1))
		break
# Prdiction with Linear Kernel
pred=np.sign(X@w+b)
conf=confusion_matrix(Y,pred,labels=[-1,1])

training_accuracy=(np.trace(conf)/np.sum(conf))*100


print("Training accuracy for Linear Kernel ",training_accuracy)

# Gaussian Kernel
P_g      = GaussianKernel()
sol_g    = solvers.qp(P_g,q,G,h,A,b)
alphas_g = np.array(sol_g['x'])

sv_idx=getSupportVectorIndices(alphas_g,1e-4)
bias_g=0
for j in sv_idx:
	if alphas_g[j]<C:
		tp=X-X[j].reshape(n,1)
		temp= float(np.sum(Y[i]*alphas_g*np.exp(-gamma*np.diagonal(tp@(tp.T)))))
		bias_g = Y[j] - temp
		break






