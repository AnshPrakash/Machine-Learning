import sys
import numpy as np
from math import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation


filex=open("./data/weightedX.csv","r")
filey=open("./data/weightedY.csv","r")



Y=np.array([float(y) for y in filey])
X=np.array([float(x) for x in filex])
Y=Y.reshape((np.shape(Y)[0]),1)
X=X.reshape((np.shape(X)[0],1))
x_std=np.std(X)+0.001
x_mean=np.mean(X)+0.001
X=(X-x_mean)*(1/x_std)
X=np.hstack((np.ones(np.shape(X)[0]).reshape((np.shape(X)[0],1)),X))


def cost(theta0,theta1):
	theta=np.array([(theta0),(theta1)]).reshape((2,1))
	tp=(X.dot(theta)-Y)
	cost=(tp.T).dot(tp)
	return(float(cost)/(2*np.shape(X)[0]))



def BGD(X,labels):
	theta=np.zeros(np.shape(X)[1]).reshape((np.shape(X)[1],1))
	diff=10*np.ones(np.shape(X)[1]).reshape((np.shape(X)[1],1))
	epsilon=0.000001
	diff=float("inf")
	learining_rate=0.1
	m=(np.shape(X)[0])
	pcost=cost(theta[0],theta[1])
	while diff>epsilon:
		temp=(X.T).dot(labels-X.dot(theta))
		theta=theta+(learining_rate/m)*temp
		new_cost=cost(theta[0],theta[1])
		diff=abs(new_cost- pcost)
		pcost=new_cost
	return(theta)



# Given training set and input x It returns the corresponding Diagonal weight Matrix
def weightMatrix(x,X):
	m=np.shape(X)[0]
	t=0.8 ##Std deiviation
	W=[float(((x-X[i].reshape(np.shape(X)[1],1)).T).dot((x-X[i].reshape(np.shape(X)[1],1)))) for i in range((m))]
	W=np.array(W)
	W=(W*(-1))/(2*t*t)
	temp=[[0]*m for _ in range(m)]
	for i in range(m):
		temp[i][i]=np.exp(W[i])
	return(np.array(temp))


# Analytical solution
def wBGD(X,Y,x):
	W=weightMatrix(x,X)
	theta=((((np.linalg.inv(((X.T).dot(W)).dot(X))).dot(X.T)).dot(W.T+W)).dot(Y))*0.5
	return(theta)


############  PLOTS FROM HERE ############################

# 2Part(a)
theta_un_weighted=BGD(X,Y)
print(theta_un_weighted)
pltx=np.linspace(-5,5,50).reshape((50,1))
pltx=np.hstack((np.ones(np.shape(pltx)[0]).reshape((np.shape(pltx)[0],1)),pltx))
plty=np.array([(theta_un_weighted.T).dot(pltx[i]) for i in range(np.shape(pltx)[0])])

fig1, ax1 = plt.subplots()
ax1.plot(pltx[:,1],plty,'-')
ax1.plot(X[:,1],Y,'o')
ax1.legend(['hypothesis','Original Data'], loc='upper left')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
plt.title('Hypothesis function(Linear regression)')


# Part(b)
fig2,ax2=plt.subplots()
pltx=np.linspace(-5,5,50).reshape((50,1))
pltx=np.hstack((np.ones(np.shape(pltx)[0]).reshape((np.shape(pltx)[0],1)),pltx))
plty=np.array([(wBGD(X,Y,pltx[i].reshape(2,1)).T).dot(pltx[i]) for i in range(np.shape(pltx)[0])])

ax2.plot(pltx[:,1],plty,'-')
ax2.plot(X[:,1],Y,'o')
ax2.legend(['hypothesis','Original Data'], loc='upper left')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
plt.title('Hypothesis function(Locally Weighted Linear regression)')
plt.show()


