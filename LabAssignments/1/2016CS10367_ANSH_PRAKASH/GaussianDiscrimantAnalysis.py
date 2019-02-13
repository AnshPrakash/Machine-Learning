import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
import math
import sys



# filex=open("./data/q4x.dat","r")
# filey=open("./data/q4y.dat","r")

filex=open(sys.argv[2],"r")
filey=open(sys.argv[3],"r")
which_to_run=int(sys.argv[4])

Y=np.array([1 if y.strip()=="Alaska" else 0 for y in filey])
Y=Y.reshape(np.shape(Y)[0],1)
X=np.array([list(map(float,(x.strip().split()))) for x in filex])
# x_std=np.std(X,axis=0)+0.001
# x_mean=np.mean(X,axis=0)+0.001
# X=(X-x_mean)*(1/x_std)

####Shuffluing Data ###################
data=np.hstack((X,Y))
np.random.shuffle(data)
X=(data[:,0:-1])
Y=data[:,-1].reshape(np.shape(X)[0],1)
Y=Y.astype(int)
# print(X[(Y==0).reshape(np.shape(Y)[0]),:])
myu0=np.mean(X[(Y==0).reshape(np.shape(Y)[0]),:],axis=0).reshape(1,np.shape(X)[1])
myu1=np.mean(X[(Y==1).reshape(np.shape(Y)[0]),:],axis=0).reshape(1,np.shape(X)[1])
print("myu0")
print(myu0)
print("myu1")
print(myu1)
n=np.shape(X)[1]

phi=np.mean(Y)
# Assuming both covariance matrices are not equal
covariance_mat0=np.zeros((n,n))
covariance_mat1=np.zeros((n,n))
X1=X[(Y==1).reshape(np.shape(Y)[0]),:]-myu1.reshape(1,np.shape(X)[1])
X0=X[(Y==0).reshape(np.shape(Y)[0]),:]-myu0.reshape(1,np.shape(X)[1])
for i in range(np.shape(X1)[0]):
	xi=X1[i].reshape(np.shape(X)[1],1)
	covariance_mat1=covariance_mat1+xi.dot(xi.T)


for i in range(np.shape(X0)[0]):
	xi=X0[i].reshape(np.shape(X)[1],1)
	covariance_mat0=covariance_mat0+xi.dot(xi.T)

''' Part(a and d)

Covariance matrices 

'''
# Assuming both are covariance matrices are equal
covariance_mat=(covariance_mat0+covariance_mat1)*(1/np.shape(X)[0])
covariance_mat0=covariance_mat0*(1/np.shape(X0)[0])
covariance_mat1=covariance_mat1*(1/np.shape(X1)[0])
print("E0")
print(covariance_mat0)

print("E1")
print(covariance_mat1)

print("E when E0=E1")
print(covariance_mat)





inv_cov=np.linalg.pinv(covariance_mat)
def exp_term(x):
	# probability y=1
	global myu0
	global myu1
	global phi
	global inv_cov
	t0=((x- myu0.reshape(np.shape(x)[0],1)).T).dot(inv_cov).dot(x- myu0.reshape(np.shape(x)[0],1))
	t1=((x- myu1.reshape(np.shape(x)[0],1)).T).dot(inv_cov).dot(x- myu1.reshape(np.shape(x)[0],1))
	return(t0-t1)





c0=np.linalg.det(covariance_mat0)
c1=np.linalg.det(covariance_mat1)
a=np.linalg.pinv(covariance_mat1)-np.linalg.pinv(covariance_mat0)
b=-2*((myu1.reshape(np.shape(X)[1],1).T).dot(np.linalg.pinv(covariance_mat1))-(myu0.reshape(np.shape(X)[1],1).T).dot(np.linalg.pinv(covariance_mat0)))
d=-1*(myu0.reshape(np.shape(X)[1],1).T).dot(np.linalg.pinv(covariance_mat0)).dot(myu0.reshape(np.shape(X)[1],1))+(myu1.reshape(np.shape(X)[1],1).T).dot(np.linalg.pinv(covariance_mat1)).dot(myu1.reshape(np.shape(X)[1],1))-math.log((1-phi)/phi)


def exp_term2(x):
	global a,b,c,d
	t=(x.T).dot(a).dot(x)+b.dot(x)+d
	return(float(t))


###############Plots ############################


# Part(b and c)
''' Plots'''
steps =380
pltx1=np.linspace(60,200,steps).reshape((steps,1))
pltx2=np.linspace(300,500,steps).reshape((steps,1))
pltX1_mesh,pltX2_mesh=np.meshgrid(pltx1,pltx2)
grid=np.zeros((steps,steps))
approx=0.01

for i in range(steps):
	for j in range(steps):
		x1 = pltX1_mesh[i][j]
		x2 = pltX2_mesh[i][j]
		x=np.array([x1,x2]).reshape(np.shape(X)[1],1)
		grid[i][j] = abs(exp_term(x))
x1=pltX1_mesh[(grid<approx)]
x2=pltX2_mesh[(grid<approx)]
grid=grid[(grid<approx)]
# print(grid)
fig1, ax1 = plt.subplots()
ax1.scatter(X[:,0],X[:,1],c=Y.ravel())
ax1.plot(x1,x2,'-',label="E0=E1")
ax1.set_xlabel('Fresh Water')
ax1.set_ylabel('Marine Water')
ax1.set_title('GDA Decsion Boundaries')


# Part(d) already done Above with part(a)
# Part(e)
if which_to_run==1:
	approx2=0.01
	grid2=np.zeros((steps,steps))
	for i in range(steps):
		for j in range(steps):
			x1 = pltX1_mesh[i][j]
			x2 = pltX2_mesh[i][j]
			x=np.array([x1,x2]).reshape(np.shape(X)[1],1)
			grid2[i][j] = abs(exp_term2(x))
	xq1=pltX1_mesh[(grid2<approx2)]
	xq2=pltX2_mesh[(grid2<approx2)]
	ax1.plot(xq1,xq2,'-',label="E0!=E1")
ax1.legend(loc='upper left')

plt.show()