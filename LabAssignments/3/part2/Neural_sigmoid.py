import pandas as pd
import numpy as np
import sys
from pprint import pprint


def sigmoid(z):
	return(1/(1+np.exp(-z)))

def sig_derivative(z):
	return(sigmoid(z)*(1-sigmoid(z)))




def forward_propogation(parameters,input):
	'''
		parameters is a dictionary containing parameters  of the neural nets
		Input is a column vector

		returns the list output labels in the range of [0-1]
	'''
	X = input
	for i in range(len(parameters)):
		X = sigmoid(X@(parameters[i].T))

	return(X)

def cost(parameters,input_data,labels):
	return(np.sum(np.square(labels - forward_propogation(parameters,input_data))))

def training_neural_net(features,labels,batch_size,hidden_units,learning_rate):
	'''
		Implements SGD with batch as input
		hidden_units =[#perceptrons in layer 1,#perceptrons in layer 2,...]
		last layer will be output layer
	
		returns a dictionary containing all the learned parameters
	'''
	hidden_units.append(len(labels[0]))
	features_size = len(features[0])
	parameters = {}
	deltas = {}
	parameters[0] = np.random.rand(hidden_units[0],features_size)/1000
	deltas[0] = np.zeros((hidden_units[0],1))
	# print(parameters[0].shape)
	for layer in range(1,len(hidden_units)):
		parameters[layer] = np.random.rand(hidden_units[layer],hidden_units[layer-1])/1000
		# print(parameters[layer].shape)
		deltas[layer] = []
		# deltas[layer] = np.zeros((hidden_units[layer],1))

	X =[0]*(len(parameters)+1)
	# Stochastic Gradient Descent with batch size
	iters = int(len(features)/batch_size)
	epochs = 100
	for _ in range(epochs):
		for i in range(iters):
			X[0] = features[i*batch_size:batch_size*(i+1)]
			labels_buf = labels[i*batch_size:batch_size*(i+1)]
			O=[0]*len(parameters)
			for j in range(len(parameters)):
				X[j+1] = sigmoid(X[j]@(parameters[j].T))


			# backpropogation
			deltas[len(parameters)-1] = (labels_buf - X[-1])
			O[-1] = deltas[len(parameters) - 1]
			for k in range(len(parameters)-2,-1,-1):
				o = sig_derivative(X[k+1]) #X is shifted because X[0] have the inputs
				O[k] = o
				deltas[k] = (deltas[k+1]@parameters[k+1])*o
				
			# parameter updates
			parameters[0]=parameters[0] + learning_rate*((deltas[0].T)@(X[0]))
			for j in range(1,len(parameters)):
				parameters[j]=parameters[j] + learning_rate*((deltas[j].T)@O[j-1])

			print(cost(parameters,X[0],labels_buf))
	return(parameters)



test_f="OneHot_poker-hand-testing.data"
train_f="OneHot_poker-hand-training-true.data"

with open(train_f) as f:
	df = pd.read_csv(f)

labels = np.array(df.iloc[:,-10:])
features = np.array(df.iloc[:,:-10])


hidden_units=[2,5,2,6]
batch_size = 100
learning_rate = 0.01
parameters= training_neural_net(features,labels,batch_size,hidden_units,learning_rate)

res = forward_propogation(parameters,features)
print(res)
pd.DataFrame(res).to_csv("check.csv")
