# python3
import time 
import numpy as np
import pandas as pd
from cvxopt import matrix
from cvxopt import solvers
from sklearn.metrics import confusion_matrix
import sys


test_f =sys.argv[2]
train_f=sys.argv[1]
start_time=time.time()
with open(train_f) as f:
	df = pd.read_csv(f,header=None)

# df=df.sample(n=1000)
M=len(df)
df.iloc[:,:-1]=df.iloc[:,:-1]/255

Y_train = np.array(df.iloc[:,-1]).reshape(M,1)
X_train = np.array(df.iloc[:,:-1])



gamma=0.05
def GaussianKernel(Y,X):
	Z=X@(X.T)
	m=len(Y)
	P=np.zeros((m,m))
	for i in range(m):
		for j in range(m):
			P[i][j]=Z[i][i]+Z[j][j]-2*Z[i][j]
	P=matrix(Y@(Y.T)*np.exp(-gamma*P),tc='d')
	return(P)

def getSupportVectorIndices(alphas,e=1e-4):
	vec=[]
	for i in range(len(alphas)):
		if abs(alphas[i])>e:
			vec.append(i)
	return(vec)

def get_training_data(i,j):
	global df
	df1=(df.loc[(df.iloc[:,-1]==i) | (df.iloc[:,-1]==j)])
	df1.iloc[:,-1]=df1.iloc[:,-1].replace({i:1,j:-1})
	Y_p = np.array(df1.iloc[:,-1]).reshape(len(df1),1)
	X_p = np.array(df1.iloc[:,:-1])
	return(Y_p,X_p)

def get_trained_model(i,j,C):
	gamma=0.05 # given in assignment (Fixed)
	Y_p,X_p = get_training_data(i,j)
	m=len(Y_p)
	P = GaussianKernel(Y_p,X_p)
	q = matrix(-1*np.ones((m,1)),tc='d')
	G = matrix(np.vstack((-np.identity(m),np.identity(m))),tc='d')
	A = matrix(Y_p.T,tc='d')
	h = matrix(np.vstack((np.zeros((m,1)),C*np.ones((m,1)))),tc='d')
	b = matrix(np.zeros((1,1)))
	sol    = solvers.qp(P,q,G,h,A,b)
	alphas = np.array(sol['x'])
	sv_idx=getSupportVectorIndices(alphas,1e-4)
	bias_g=0
	count_g=1
	temp_XXT=X_p@(X_p.T)
	temp_xxt_d=np.diagonal(temp_XXT).reshape(m,1)
	# taking mean over support_vectors
	buf= 10 if len(sv_idx)>10 else len(sv_idx)
	for i in sv_idx[:buf]:
		if alphas[i]<C:
			bias_g+=Y_p[i]-np.sum(Y_p*alphas*(np.exp(-gamma*(temp_xxt_d +temp_xxt_d[i] +temp_XXT[:,i].reshape(m,1)))))
			count_g+=1
	bias_g=bias_g/count_g
	X_sv = X_p[sv_idx]
	Y_sv = Y_p[sv_idx]
	alphas_sv=alphas[sv_idx]
	model=[X_sv,Y_sv,alphas_sv,bias_g]
	return(model)




def training_models(C):
	# creatin 45 separate classification problems
	models   = {}
	for i in range(9):#0-8
		for j in range(i+1,10): #every git grater than this one
			models[(i,j)]=get_trained_model(i,j,C)
	return(models)



def prediction_uni(Z,model):
	# Z is Test set m2 X n numpy array
	[X,Y,alphas_g,bias_g]=model
	x_d =np.diagonal(X@(X.T)).reshape(len(X),1)
	z_d =np.diagonal(Z@(Z.T)).reshape(len(Z),1)
	S =-2*X@(Z.T)
	m=len(S)
	tp=alphas_g*Y
	pred=[0]*len(Z)
	for i in range(len(Z)):
		pred[i]=float(np.sign(np.sum(tp*(np.exp(-gamma*(x_d+z_d[i]+S[:,i].reshape(m,1)))))+bias_g))
	return(pred)

def prediction_multi(X,Y_labels,models):
	# Calculating training Set Accuracy
	m = len(Y_labels)
	wins = np.zeros((m,10),dtype=int)
	for i in range(9):
		for j in range(i+1,10):
			results=prediction_uni(X,models[(i,j)])
			for k in range(len(results)):
				if results[k]==1:
					wins[k][i] +=1
				else:
					wins[k][j] +=1
	preds=len(wins[0]) - np.argmax(np.flip(wins,1),axis=1)-1
	return(preds)

models=training_models(1.0)
preds=prediction_multi(X_train,Y_train,models)
print("Training set Accuracy :",np.mean(preds==Y_train))
conf=confusion_matrix(Y_train,preds,labels=[0,1,2,3,4,5,6,7,8,9])
print(conf)

# Test Set accuracy
with open(test_f) as f:
	df1 = pd.read_csv(f,header=None)

# df1=df1.sample(n=100)

m_test=len(df1)
df1.iloc[:,:-1]=df1.iloc[:,:-1]/255

Y_test = np.array(df1.iloc[:,-1]).reshape(m_test,1)
X_test = np.array(df1.iloc[:,:-1])


preds=prediction_multi(X_test,Y_test,models)
print("Test Set Accuracy: ",np.mean(preds==Y_test))
# print(Y_test,preds)
conf=confusion_matrix(Y_test,preds,labels=[0,1,2,3,4,5,6,7,8,9])
print(conf)


end_time=time.time()
print("Time taken :",end_time- start_time)