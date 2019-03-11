# Python 3
import time 
import numpy as np
import pandas as pd
from cvxopt import matrix
from cvxopt import solvers
from sklearn.metrics import confusion_matrix

start_time=time.time()
# get the support vector
entry_no=7
start_time=time.time()
with open('./mnist/train.csv') as f:
	df = pd.read_csv(f,header=None)

df1=(df.loc[(df.iloc[:,-1]==entry_no) | (df.iloc[:,-1]==(entry_no+1)%10)])

df1.iloc[:,-1]=df1.iloc[:,-1].replace({entry_no:1,(entry_no+1)%10:-1})
df1.iloc[:,:-1]=df1.iloc[:,:-1]/255
Y = np.array(df1.iloc[:,-1])
X = np.array(df1.iloc[:,:-1])
m=len(Y)
C = 1.0
Y=Y.reshape(m,1)
n=28*28 #28 X 28 dimension images

def getSupportVectorIndices(alphas,e=1e-4):
	vec=[]
	for i in range(m):
		if abs(alphas[i])>e:
			vec.append(i)
	return(vec)

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


sv=getSupportVectorIndices(alphas,1e-7)

w = (X.T).dot(alphas*Y)


bias=np.mean(Y[sv]-X[sv]@w)
# Prdiction with Linear Kernel
pred=np.sign(X@w+bias)
conf=confusion_matrix(Y,pred,labels=[-1,1])

training_accuracy=(np.trace(conf)/np.sum(conf))*100


print("Training accuracy for Linear Kernel ",training_accuracy)

# Gaussian Kernel
P_g      = GaussianKernel()
sol_g    = solvers.qp(P_g,q,G,h,A,b)
alphas_g = np.array(sol_g['x'])

sv_idx=getSupportVectorIndices(alphas_g,1e-4)


bias_g=0
count_g=1
temp_XXT=X@(X.T)
temp_xxt_d=np.diagonal(temp_XXT).reshape(m,1)
# taking mean over support_vectors
buf= 10 if len(sv_idx)>10 else len(sv_idx)

for i in sv_idx[:buf]:
	if alphas_g[i]<C:
		bias_g+=Y[i]-np.sum(Y*alphas_g*(np.exp(-gamma*(temp_xxt_d +temp_xxt_d[i] +temp_XXT[:,i].reshape(m,1)))))
		count_g+=1
bias_g=bias_g/count_g





def accuracy_gaussian(Z,y_labels):
	# Z is Test set m2 X n numpy array
	x_d =np.diagonal(X@(X.T)).reshape(len(X),1)
	z_d =np.diagonal(Z@(Z.T)).reshape(len(Z),1)
	S =-2*X@(Z.T)
	tp=alphas_g*Y
	pred=[0]*len(Z)
	for i in range(len(Z)):
		pred[i]=np.sign(np.sum(tp*(np.exp(-gamma*(x_d+z_d[i]+S[:,i].reshape(m,1)))))+bias_g)
	conf_g=confusion_matrix(y_labels,pred,labels=[-1,1])
	print(conf_g)
	accuracy=(np.trace(conf_g)/np.sum(conf_g))*100
	print("Accuracy of the set : ",accuracy)
	return(accuracy)

print("Training set accuracy by Gaussian Kernel",accuracy_gaussian(X,Y))



# Test Data
with open('./mnist/test.csv') as f:
	df_test = pd.read_csv(f,header=None)

df1_test=(df_test.loc[(df_test.iloc[:,-1]==entry_no) | (df_test.iloc[:,-1]==(entry_no+1)%10)])

df1_test.iloc[:,-1]=df1_test.iloc[:,-1].replace({entry_no:1,(entry_no+1)%10:-1})
df1_test.iloc[:,:-1]=df1_test.iloc[:,:-1]/255
Y_test = np.array(df1_test.iloc[:,-1])
X_test = np.array(df1_test.iloc[:,:-1])
m_test=len(Y_test)
Y_test=Y_test.reshape(m_test,1)


conf_test=confusion_matrix(Y_test,np.sign(X_test@w+bias),labels=[-1,1])

test_accuracy=(np.trace(conf_test)/np.sum(conf_test))*100
print("test set accuracy with Linear Kernel: ",test_accuracy)

print("Test set accuracy with GaussianKernel: ",accuracy_gaussian(X_test,Y_test))

end_time=time.time()
print("Time Taken ",end_time- start_time)

			



