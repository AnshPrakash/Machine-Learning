import pandas as pd
import numpy as np
from math import log
from pprint import pprint
from itertools import repeat
from sklearn.metrics import confusion_matrix
import sys

values_set = {}
data = []
attributes = []

def selectBestAttribute(indices):
	'''
		select the best attribute to split from the data 
		return a sting from attributes available
	'''
	ig  = list(map(getInformationGain,repeat(indices),attributes))
	idx = np.argmax(ig)
	return(attributes[idx],ig[idx])
	# return(attributes[np.argmax(list(map(getInformationGain,repeat(indices),attributes)))])

	

def split(attribute,indices):
	'''
		Based on the given attribute split the data 
		and return the data frames
	'''
	splited_data={}
	data_tp = data.iloc[indices,:]
	for val in values_set[attribute] :
		splited_data[val] = (data_tp.index[data_tp[attribute] == val])
	return(splited_data)

def classify(indices):
	'''
		returns the classified label
	'''
	data_temp = data.iloc[indices,:]
	p = sum(data_temp['Y'] ==1)/len(indices)
	if p >0.5 :
		return(1)
	else: 
		return(0)

def getInformationGain(indices,attribute):
	'''
		returns the information gain by splitiing  
		the data on the given attribute
	'''
	data_tp = data.iloc[indices,:]
	total = len(data_tp)
	Py1 = sum(data_tp['Y']==1)/total
	#  py1 == 1 or 0 we don't need to calculate this
	# print(sum(data['Y']==1))
	# print(data_tp['Y']==1)
	# print(data_tp)
	hy = -(Py1*log(Py1,2) + (1-Py1)*log((1-Py1),2))
	hy_x = 0.0
	for val in values_set[attribute]:
		# print(data[attribute] == values_set[attribute][val])
		df_temp = data_tp[data_tp[attribute] == val]
		px = len(df_temp)/total
		if px != 0:
			py1_x = sum(df_temp['Y'] == 1)/len(df_temp)
			if py1_x == 0 or py1_x==1:
				pass
			else:
				hy_x -= px*(py1_x*log(py1_x,2) + (1-py1_x)*log((1-py1_x),2))

	informationGain = hy -hy_x
	return(informationGain)


def checkDataPure(indices):
	'''	
		check if the data is pure or not
	'''
	labels = data.iloc[indices,-1]
	return(False if len(set(labels)) > 1 else True )



def prediction(tree,input):
	if isinstance(tree[1][input[tree[0]]],int):
		return(tree[1][input[tree[0]]])
	else:
		return(prediction(tree[1][input[tree[0]]],input))


def buildDecisionTree(indices):
	'''
		
		returns a Dictionary(which actually is a tree)
	'''
	tree = {}
	# print(len(indices))
	# print(indices)
	if checkDataPure(indices) or len(indices)<=10:
		# print("bye1")
		return(classify(indices))
	# print("Hello1")
	attribute,ig = selectBestAttribute(indices)
	if ig == 0:
		# print("bye3")
		return(classify(indices))
	splited_data =split(attribute,indices)
	temp = {}
	for val in splited_data:
		# print(val)
		if len(splited_data[val]) == 0:
			# print("bye2")
			temp[val] = classify(indices) #No occueence of this value in the data given
		else:
			temp[val] = buildDecisionTree(splited_data[val])
	tree = [attribute,temp]
	# print("bye3")
	return(tree)


test_f="credit-cards.test.csv"
validation_f = "credit-cards.val.csv"
train_f = "credit-cards.train.csv"
train_f = "processesd_credit-cards.train.csv"
with open(train_f) as f:
	df = pd.read_csv(f)

data = (df.iloc[:,2:]) ## contains labels and data both (don't need X0)
labels = df.iloc[:,-1]
attributes = list(df.columns.values)
attributes = attributes[2:-1] # contains attribute from X1 to X23
del df 
for attribute in attributes:
	values_set[attribute] =list(set(data[attribute]))

tree = buildDecisionTree(data.index[[True]*len(data)])


# pprint(tree)
pred =[]
for i in range(len(data)):
	inp = data.iloc[i]
	pred.append(prediction(tree,inp))


conf_mat = confusion_matrix(list(data.iloc[:,-1]),pred,labels=[0,1])
print("accuracy ",(np.sum(np.diagonal(conf_mat))/np.sum(conf_mat))*100,"%")
print(conf_mat)

# pprint(values_set)
