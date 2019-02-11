import sys
import numpy as np
from math import *



filex=open("./data/logisticX.csv","r")
filey=open("./data/logisticY.csv","r")


Y=np.array([float(y) for y in filey])
X=np.array([list(map(float,(x.split(",")))) for x in filex])
Y=Y.reshape((np.shape(Y)[0]),1)
X=X.reshape((np.shape(X)[0],len(X[0])))
X=np.hstack((np.ones(np.shape(X)[0]).reshape((np.shape(X)[0],1)),X))


def sigmoid(x):
	return(1/(1+np.exp(-x)))


def NewtonMethod(X,Y):
	theta=np.random.rand(np.shape(X)[1],1)
	epsilon=0.00001
	delta=float('inf')
	iter=0
	while delta>epsilon:
		tp=X.dot(theta)
		sigv=sigmoid(tp)
		print(np.shape(theta))
		g_v=(sigv)*(1.0-sigv)
		print(np.shape(g_v))
		idd=np.identity(np.shape(g_v)[0], dtype = float)
		for i in range(np.shape(g_v)[0]):
			idd[i][i]=g_v[i]
		D=idd
		try:
			hess=np.linalg.inv(((X.T).dot(D)).dot(X))
		except np.linalg.LinAlgError:
			print(iter)
			return(theta)
		diff=hess.dot(X.T).dot(Y-tp)
		theta=theta+diff
		delta=float(diff.max(axis=0))
		iter+=1
	return(theta)

print(NewtonMethod(X,Y))
