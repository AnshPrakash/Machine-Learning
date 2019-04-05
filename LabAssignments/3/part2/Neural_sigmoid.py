import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sys
from pprint import pprint


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
		X = sigmoid(X@(parameters[i].T))
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
	np.random.seed(1)
	parameters[0] = np.random.rand(hidden_units[0],features_size)
	deltas[0] = np.zeros((hidden_units[0],1))
	# print(parameters[0].shape)
	for layer in range(1,len(hidden_units)):
		parameters[layer] = np.random.rand(hidden_units[layer],hidden_units[layer-1])
		# print(parameters[layer].shape)
		deltas[layer] = []
		# deltas[layer] = np.zeros((hidden_units[layer],1))
	print("Initial Parameters")
	pprint(parameters)
	X =[0]*(len(parameters)+1)
	# Stochastic Gradient Descent with batch size
	iters = int(len(features)/batch_size)
	epochs = 1000
	for _ in range(epochs):
		for i in range(iters):
			X[0] = features[i*batch_size:batch_size*(i+1)]
			labels_buf = labels[i*batch_size:batch_size*(i+1)]
			O=[0]*len(parameters)
			for j in range(len(parameters)):
				X[j+1] = sigmoid(X[j]@(parameters[j].T))


			# backpropogation
			O[-1] = sig_derivative(X[-1])
			deltas[len(parameters)-1] = -(labels_buf - X[-1])*O[-1]
			# O[-1] = deltas[len(parameters) - 1]
			for k in range(len(parameters)-2,-1,-1):
				O[k] = sig_derivative(X[k+1]) #X is shifted because X[0] have the inputs
				deltas[k] = (deltas[k+1]@parameters[k+1])*O[k]
				
			# parameter updates
			# parameters[0]=parameters[0] - learning_rate*((deltas[0].T)@(X[0]))
			for j in range(len(parameters)):
				parameters[j]=parameters[j] - learning_rate*((deltas[j].T)@X[j])

		print(cost(parameters,X[0],labels_buf))
	return(parameters)



test_f="OneHot_poker-hand-testing.data"
train_f="OneHot_poker-hand-training-true.data"

with open(train_f) as f:
	df = pd.read_csv(f)

# with open(test_f) as f:
# 	df = pd.read_csv(f)



labels = np.array(df.iloc[:,-10:])
features = np.array(df.iloc[:,:-10])


hidden_units  = [25]
batch_size    =  25
learning_rate = 0.1
parameters    = training_neural_net(features,labels,batch_size,hidden_units,learning_rate)

# parameters = {}
# parameters[0] = np.random.rand(hidden_units[0],len(features[0]))
# for layer in range(1,len(hidden_units)):
# 	parameters[layer] = np.random.rand(hidden_units[layer],hidden_units[layer-1])
print("Trained Parameters")	
pprint(parameters)
res  = forward_propogation(parameters,features)
pred = prediction(res)
print("Accuracy is ",accuracy(pred,np.argmax(labels,axis=1))*100,"%")
conf_mat = confusion_matrix(np.argmax(labels,axis=1),pred,labels=[0,1,2,3,4,5,6,7,8,9])
print(conf_mat)

pd.DataFrame(pred).to_csv("check1.csv")
pd.DataFrame(res).to_csv("check.csv")
