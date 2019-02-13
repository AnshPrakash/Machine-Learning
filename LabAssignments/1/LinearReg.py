import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
import time
plt.style.use('seaborn-white')


interval=0.2
filex=open("./data/linearX.csv","r")
filey=open("./data/linearY.csv","r")

labels=np.array([float(y) for y in filey])
X=np.array([float(x) for x in filex])
labels=labels.reshape((np.shape(labels)[0]),1)
X=X.reshape((np.shape(X)[0],1))
x_std=np.std(X)+0.001
x_mean=np.mean(X)+0.001
X=(X-x_mean)*(1/x_std)
# y_std=np.std(labels)+0.001
# y_mean=np.mean(labels)+0.001
# labels=(labels-y_mean)*(1/(y_mean))
X=np.hstack((np.ones(np.shape(X)[0]).reshape((np.shape(X)[0],1)),X))
# print(X)

def cost(theta0,theta1):
	theta=np.array([(theta0),(theta1)]).reshape((2,1))
	tp=(X.dot(theta)-labels)
	cost=(tp.T).dot(tp)
	return(float(cost)/(2*np.shape(X)[0]))




def BGD(X,labels):
	theta=np.zeros(np.shape(X)[1]).reshape((np.shape(X)[1],1))
	diff=10*np.ones(np.shape(X)[1]).reshape((np.shape(X)[1],1))
	epsilon=0.00000001
	diff=float("inf")
	learining_rate=0.1
	m=(np.shape(X)[0])
	pcost=cost(theta[0],theta[1])
	mem=[(theta,pcost)]
	iter=0
	while diff>epsilon:
		temp=(X.T).dot(labels-X.dot(theta))
		theta=theta+(learining_rate/m)*temp
		new_cost=cost(theta[0],theta[1])
		diff=abs(new_cost- pcost)
		pcost=new_cost
		mem.append((theta,pcost))
	return(theta,mem)


theta,mem=BGD(X,labels)
print(cost(theta[0],theta[1]))
print(theta)
############  PLOTS FROM HERE ############################

# Part(a)
pltx=np.linspace(-5,5,50).reshape((50,1))
pltx=np.hstack((np.ones(np.shape(pltx)[0]).reshape((np.shape(pltx)[0],1)),pltx))
plty=np.array([(theta.T).dot(pltx[i]) for i in range(np.shape(pltx)[0])])

fig1, ax1 = plt.subplots()
ax1.plot(pltx[:,1],plty,'-')
ax1.plot(X[:,1],labels,'o')
ax1.legend(['hypothesis','Original Data'], loc='upper left')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
plt.title('Hypothesis function')

# Part(b)
steps=50
theta0=np.linspace(-2,2,steps).reshape((steps,1))
theta1=np.linspace(-2,2,steps).reshape((steps,1))
th0_mesh,th1_mesh=np.meshgrid(theta0,theta1)
H=[[0]*steps for _ in range(steps)]

for i in range(steps):
	for j in range(steps):
		H[i][j]=cost(th0_mesh[i][j],th1_mesh[i][j])
H=np.array(H)



fig2=plt.figure()
ax2 = fig2.add_subplot(111)
ax2.set_xlabel('theta0')
ax2.set_ylabel('theta1')
ax2.set_title('COST FUNCTION CONTOURS')
ax2.contour(th0_mesh,th1_mesh,H, colors='blue')

plt.ion()
for i in range(len(mem)):
	plt.pause(interval)
	ax2.plot([float(mem[i][0][0])],[float(mem[i][0][1])],'^',color='r')
	# iter+=1

# plt.show()
plt.ioff()
fig3 = plt.figure()
ax = fig3.add_subplot(111,projection='3d')
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('Cost function J(ϴ)')
ax.set_title('COST FUNCTION')
ax.plot_surface(th0_mesh,th1_mesh,H, rstride=1, cstride=1, edgecolor='none')
# plt.plot(x,y,z,'o')
plt.ion()
for i in range(len(mem)):
	# x.append(float(mem[i][0][0]))
	# y.append(float(mem[i][0][1]))
	# z.append(mem[i][1])
	plt.pause(interval)
	ax.plot([float(mem[i][0][0])],[float(mem[i][0][1])],mem[i][1],'^',color='r')



		
# ani=animation.FuncAnimation(fig,animate,fargs=(mem),interval=2000)

# plt.plot(x,y,'^',color='b')
plt.ioff()
# plt.show()
