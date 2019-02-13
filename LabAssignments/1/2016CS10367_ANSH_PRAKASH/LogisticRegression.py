import sys
import numpy as np
from math import *
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation



# filex=open("./data/logisticX.csv","r")
# filey=open("./data/logisticY.csv","r")


filex=open(sys.argv[2],"r")
filey=open(sys.argv[3],"r")

Y=np.array([float(y) for y in filey])
X=np.array([list(map(float,(x.split(",")))) for x in filex])
Y=Y.reshape((np.shape(Y)[0]),1)
X=X.reshape((np.shape(X)[0],len(X[0])))
x_std=np.std(X,axis=0)+0.01
x_mean=np.mean(X,axis=0)+0.01
X=(X-x_mean)*(1/x_std)
X=np.hstack((np.ones(np.shape(X)[0]).reshape((np.shape(X)[0],1)),X))
data=np.hstack((X,Y))
np.random.shuffle(data)
X=(data[:,0:-1])
Y=data[:,-1].reshape(100,1)
# print(np.shape(Y))
# print((X))


def sigmoid(x):
	return(1/(1+np.exp(-x)))


def NewtonMethod(X,Y):
	theta=np.ones((np.shape(X)[1],1))
	epsilon=0.0000001
	delta=float('inf')
	iter=0
	while delta>epsilon:
		# print("hello")
		# print(theta)
		tp=X.dot(theta)
		sigv=sigmoid(tp)
		g_v=(sigv)*(1.0-sigv)
		idd=np.zeros((np.shape(g_v)[0],np.shape(g_v)[0]))
		for i in range(np.shape(g_v)[0]):
			idd[i][i]=g_v[i]
		D=idd
		try:
			hess=np.linalg.pinv(((X.T).dot(D)).dot(X))
		except np.linalg.LinAlgError:
			print("Singular Hessian")
			return(theta)
		diff=hess.dot(X.T).dot(Y-sigv)
		theta=theta+diff
		delta=float(np.abs(diff).max(axis=0))
		iter+=1
	return(theta)

theta=(NewtonMethod(X,Y))
print(theta)
#############PLOTING STARTS FROM HERE##########################
# Decision Boundary
colors=["red","blue"]
fig1, ax1 = plt.subplots()
steps =100
pltx1=np.linspace(-5,5,steps).reshape((steps,1))
pltx2=np.linspace(-5,5,steps).reshape((steps,1))
pltX1_mesh,pltX2_mesh=np.meshgrid(pltx1,pltx2)
grid=np.zeros((steps,steps))
approx=0.01

for i in range(steps):
	for j in range(steps):
		x1 = pltX1_mesh[i][j]
		x2 = pltX2_mesh[i][j]
		x=np.array([1,x1,x2]).reshape(3,1)
		grid[i][j] = abs(float((theta.T).dot(x)))

x1=pltX1_mesh[(grid<approx)]
x2=pltX2_mesh[(grid<approx)]
grid=grid[(grid<approx)]
ax1.scatter(X[:,1],X[:,2],c=Y.ravel())
ax1.plot(x1,x2,'-')

##################need to Find Decision Boundary



ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
plt.title('Decision Boundary(Logistic Regression)')
plt.show()

