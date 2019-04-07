# Q2 c
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sys
from pprint import pprint
import matplotlib.pyplot as plt


def sigmoid(z):
	return(1/(1+np.exp(-z)))

def sig_derivative(z):
	q = sigmoid(z)
	return(q*(1-q))

def accuracy(pred,labels):
	return(np.sum(pred == labels)/len(labels))


def forward_propogation(parameters,input):
	'''
		p3,arameters is a dictionary containing parameters  of the neural nets
		Input is a column vector

		returns the list output labels in the range of [0-1]
	'''
	X = input
	for i in range(len(parameters)):
		X = sigmoid(X@(parameters[i][0].T) + parameters[i][1])
	return(X)
	

def cost(parameters,input_data,labels):
	return(np.sum(np.square(labels - forward_propogation(parameters,input_data)))/len(input_data))

def prediction(pred):
	return(np.argmax(pred,axis = 1))

def training_neural_net(features,labels,batch_size,hidden_units,learning_rate):
	'''
		Implements SGD with batch as input
		hidden_units =[#perceptrons in layer 1,#perceptrons in layer 2,...]
		last layer will be output layer
	
		returns a dictionary containing all the learned parameters
	'''
	learning_rate=learning_rate/batch_size
	hidden_units.append(len(labels[0]))
	features_size = len(features[0])
	parameters = {}
	deltas = {}
	# np.random.seed(1)
	parameters[0] = [np.random.rand(hidden_units[0],features_size),np.random.rand(1,hidden_units[0])]
	deltas[0] = np.zeros((hidden_units[0],1))
	# print(parameters[0].shape)
	for layer in range(1,len(hidden_units)):
		parameters[layer] = [np.random.rand(hidden_units[layer],hidden_units[layer-1]),np.random.rand(1,hidden_units[layer])]
		# print(parameters[layer].shape)
		deltas[layer] = []
		# deltas[layer] = np.zeros((hidden_units[layer],1))
	# print("Initial Parameters")
	# pprint(parameters)
	X =[0]*(len(parameters)+1)
	# Stochastic Gradient Descent with batch size
	iters = int(len(features)/batch_size)
	epochs = 3000
	for _ in range(epochs):
		for i in range(iters):
			X[0] = features[i*batch_size:batch_size*(i+1)]
			labels_buf = labels[i*batch_size:batch_size*(i+1)]
			O=[0]*len(parameters)
			for j in range(len(parameters)):
				X[j+1] = sigmoid(X[j]@(parameters[j][0].T) + parameters[j][1])


			# backpropogation
			O[-1] = sig_derivative(X[-1])
			deltas[len(parameters)-1] = -(labels_buf - X[-1])*O[-1]
			for k in range(len(parameters)-2,-1,-1):
				O[k] = sig_derivative(X[k+1]) #X is shifted because X[0] have the inputs
				deltas[k] = (deltas[k+1]@parameters[k+1][0])*O[k]
				
			# parameter updates
			for j in range(len(parameters)):
				parameters[j][0] = parameters[j][0] - learning_rate*((deltas[j].T)@X[j])
				parameters[j][1] = parameters[j][1] - learning_rate*(np.sum(deltas[j],axis = 0 ,keepdims =True))

		# print("J :",cost(parameters,X[0],labels_buf))
	return(parameters)



test_f="OneHot_poker-hand-testing.data"
train_f="OneHot_poker-hand-training-true.data"

with open(train_f) as f:
	df = pd.read_csv(f)




labels = np.array(df.iloc[:,-10:])
features = np.array(df.iloc[:,:-10])



with open(test_f) as f:
	df = pd.read_csv(f)

labels_test = np.array(df.iloc[:,-10:])
features_test = np.array(df.iloc[:,:-10])

del df

hidden_units_l = [[5],[10],[15],[20],[25]]
batch_size    =  25
learning_rate = 0.1
# hidden_units  = [25]

# parameters = {}
# parameters[0] = [np.random.rand(hidden_units[0],len(features[0])),np.random.rand(1,hidden_units[0])]
# for layer in range(1,len(hidden_units)):
# 	parameters[layer] = [np.random.rand(hidden_units[layer],hidden_units[layer-1]),np.random.rand(1,hidden_units[layer])]

train_acc = []
test_acc = []
print("Learning Rate ",learning_rate)
for hidden_units in hidden_units_l:
	parameters    = training_neural_net(features,labels,batch_size,hidden_units,learning_rate)
	print("Trained Parameters")	
	# pprint(parameters)
	res  = forward_propogation(parameters,features)
	pred = prediction(res)
	train_acc.append(accuracy(pred,np.argmax(labels,axis=1))*100)
	# print("Accuracy is ",accuracy(pred,np.argmax(labels,axis=1))*100,"%")
	print("Hidden Units in a Single layer ",hidden_units[0])
	conf_mat = confusion_matrix(np.argmax(labels,axis=1),pred,labels=[0,1,2,3,4,5,6,7,8,9])
	print("Accuracy on Training Set ",accuracy(pred,np.argmax(labels,axis=1))*100,"%")
	print("Training Confusion Matrix")
	print(conf_mat)
	res  = forward_propogation(parameters,features_test)
	pred = prediction(res)
	test_acc.append(accuracy(pred,np.argmax(labels_test,axis=1))*100)
	conf_mat = confusion_matrix(np.argmax(labels_test,axis=1),pred,labels=[0,1,2,3,4,5,6,7,8,9])
	print("Accuracy on Test Set ",accuracy(pred,np.argmax(labels_test,axis=1))*100,"%")
	print("Confusion Matrix for Test Set")
	print(conf_mat)


hidden_units = [5,10,15,20,25]

plt.style.use("ggplot")
plt.figure()
plt.plot(hidden_units,train_acc,label="Train accuracy")
plt.plot(hidden_units,test_acc,label = "Test accuracy")
plt.ylim(0, 100)

plt.title('Accuracy Vs Hidden units in a Single Layer')
plt.xlabel("# Hidden Units")
plt.ylabel("Accuracy %")
plt.legend(loc="upper right")

plt.show()


