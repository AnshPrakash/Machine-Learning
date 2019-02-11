import sys
import numpy as np


filex=open("./data/linearX.csv","r")
filey=open("./data/linearY.csv","r")

labels=np.array([float(y) for y in filey])
X=np.array([float(x) for x in filex])
labels=labels.reshape((np.shape(labels)[0]),1)
X=X.reshape((np.shape(X)[0],1))
X=np.hstack((np.ones(np.shape(X)[0]).reshape((np.shape(X)[0],1)),X))



def cost(theta):
	tp=(X.dot(theta)-labels)
	cost=(tp.T).dot(tp)
	return(float(cost))


def BGD(X,labels):
	theta=np.zeros(np.shape(X)[1]).reshape((np.shape(X)[1],1))
	diff=10*np.ones(np.shape(X)[1]).reshape((np.shape(X)[1],1))
	epsilon=0.00001
	diff=float("inf")
	learining_rate=0.01
	m=(np.shape(X)[0])
	while diff>epsilon:
		temp=(X.T).dot(labels-X.dot(theta))
		theta=theta+(learining_rate/m)*temp
		temp=abs(temp)
		diff=float(temp.max(axis=0))*(learining_rate/m)
	return(theta)


theta=BGD(X,labels)
print(cost(theta))
# print(theta)
