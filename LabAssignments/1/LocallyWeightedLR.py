import sys
import numpy as np
from math import *


filex=open("./data/weightedX.csv","r")
filey=open("./data/weightedY.csv","r")



Y=np.array([float(y) for y in filey])
X=np.array([float(x) for x in filex])
Y=Y.reshape((np.shape(Y)[0]),1)
X=X.reshape((np.shape(X)[0],1))
X=np.hstack((np.ones(np.shape(X)[0]).reshape((np.shape(X)[0],1)),X))



# Given training set and input x It returns the corresponding Diagonal weight Matrix
def weightMatrix(x,X):
	m=np.shape(X)[0]
	t=0.8 ##Std deiviation
	W=[ ((x-X[i]).T).dot((x-X[i]).T) for i in range((m))]
	W=np.array(W)
	W=(W*(-1))/(2*t*t)
	temp=[[0]*m for _ in range(m)]
	for i in range(m):
		temp[i][i]=exp(W[i])
	return(np.array(temp))


# Analytical solution
def wBGD(X,Y,x):
	W=weightMatrix(x,X)
	theta=((((np.linalg.inv(((X.T).dot(W)).dot(X))).dot(X.T)).dot(W.T+W)).dot(Y))*0.5
	return(theta)

print(wBGD(X,Y,X[0].T))
