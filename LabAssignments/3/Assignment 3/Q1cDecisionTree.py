import pandas as pd
import numpy as np
from math import log,isnan
from pprint import pprint
from itertools import repeat
from sklearn.metrics import confusion_matrix
import sys
import operator
from functools import reduce

values_set = {}
data = []
continous = {"X1","X5","X12","X13","X14","X15","X16","X17","X18","X19","X20","X21","X22","X23" }
all_columns = ["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15"
				,"X16","X17","X18","X19","X20","X21","X22","X23","Y"]
attributes = []
num_ofnodes = 0
def selectBestAttribute(indices):
	'''
		select the best attribute to split from the data 
		return a sting from attributes available
	'''
	data_tp = data.iloc[indices,:]
	total = len(data_tp)
	Py1 = np.mean(data_tp['Y'])
	data_tp = data.iloc[indices,:]
	ig  = np.array(list(map(getInformationGain,repeat(data_tp),attributes,repeat(Py1))))
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
	if attribute in continous:
		med =  data_tp[attribute].median()
		[splited_data[0],splited_data[1]] = [data_tp.index[data_tp[attribute]<med],data_tp.index[data_tp[attribute] >= med]]
	for val in values_set[attribute]:
		splited_data[val] = (data_tp.index[data_tp[attribute] == val])
	return(splited_data)

def classify(indices):
	'''
		returns the classified label
	'''
	data_temp = data.iloc[indices,:]
	p = np.mean(data_temp['Y'])
	if p >0.5 :
		return(1)
	else: 
		return(0)

def getInformationGain(data_tp,attribute,Py1):
	'''
		returns the information gain by splitiing  
		the data on the given attribute
	'''
	y_entropy = -(Py1*(np.log2(Py1)) + (1-Py1)*np.log2(1-Py1))
	if attribute in continous:
		med =  data_tp[attribute].median()
		children = [data_tp[data_tp[attribute]<med].iloc[:,-1],data_tp[data_tp[attribute] >= med].iloc[:,-1]]
	else:	
		children =  list(map(lambda val: data_tp[data_tp[attribute]==val].iloc[:,-1] ,values_set[attribute]))
	px_children = list(map(lambda df:len(df)/len(data_tp),children))
	cond_py1x = list(map(lambda lb: np.mean(lb),children))
	entopies = list(map(lambda p: 0 if (p==0 or p==1 or isnan(p) )  else -p*np.log2(p)-(1-p)*np.log2(1-p),cond_py1x))
	avg_ent = list(map(operator.mul,entopies,px_children))
	hy_x = reduce(operator.add,avg_ent)
	informationGain = y_entropy - hy_x
	return(informationGain)


def checkDataPure(indices):
	'''	
		check if the data is pure or not
	'''
	if len(indices)>0:
		labels = data.iloc[indices,-1]
		return(np.all(labels  == labels[indices[-1]]))
	else:
		return(True)
	# return(False if len(set(labels)) > 1 else True )



def prediction(tree,input):
	val = input[tree[0]]
	if tree[0] in continous:
		med = data.iloc[tree[2]][tree[0]].median()
		val = 0 if input[tree[0]]<med else 1
	if isinstance(tree[1][val],int):
		return(tree[1][val])
	else:
		return(prediction(tree[1][val],input))	


def buildDecisionTree(indices):
	'''
		returns a Dictionary(which actually is a tree)
	'''
	tree = {}
	global num_ofnodes
	num_ofnodes += 1
	if len(indices)<=10 or checkDataPure(indices) :
		return(classify(indices))
	attribute,ig = selectBestAttribute(indices)
	# print("Best Attribute for this node is ",attribute," with information gain ",ig)
	if ig == 0:
		return(classify(indices))
	splited_data =split(attribute,indices)
	temp = {}
	for val in splited_data:
		if len(splited_data[val]) == 0:
			temp[val] = classify(indices) #No occueence of this value in the data given
		else:
			temp[val] = buildDecisionTree(splited_data[val])
	tree = [attribute,temp,indices]
	return(tree)



test_f="credit-cards.test.csv"
validation_f = "credit-cards.val.csv"
train_f = "credit-cards.train.csv"


# validation_f = "processesd_credit-cards.val.csv"
# test_f = "processesd_credit-cards.test.csv"
# train_f = "processesd_credit-cards.train.csv"

with open(train_f) as f:
	df = pd.read_csv(f)

data = (df.ix[1:,1:]) ## contains labels and data both (don't need X0)
labels = data.ix[:,-1] #df.iloc[1:,-1]
data = data.astype(int)
labels =labels.astype(int)


data =pd.DataFrame(np.array(data),columns=all_columns)
labels =pd.DataFrame(np.array(labels),columns=['Y'])

# print(data)
attributes = list(df.columns.values)
attributes = attributes[1:-1] # contains attribute from X1 to X23
del df 
for attribute in attributes:
	if attribute in continous:
		values_set[attribute] = [0,1]
	else:
		values_set[attribute] =list(set(data[attribute]))


tree = buildDecisionTree(data.index[[True]*len(data)])
print("Total Num of Nodes",num_ofnodes)

# pprint(tree)

pred =[]
for i in range(len(data)):
	inp = data.iloc[i]
	pred.append(prediction(tree,inp))


conf_mat = confusion_matrix(list(data.iloc[:,-1]),pred,labels=[0,1])
print("accuracy ",(np.sum(np.diagonal(conf_mat))/np.sum(conf_mat))*100,"%")
print(conf_mat)
# pprint(values_set)


with open(test_f) as f:
	df = pd.read_csv(f)


test_data = (df.iloc[1:,1:]) ## contains labels and data both (don't need X0)
labels_test = test_data.iloc[:,-1]
test_data = test_data.astype(int)
labels_test =labels_test.astype(int)



test_data =pd.DataFrame(np.array(test_data),columns=all_columns)
test_labels =pd.DataFrame(np.array(labels_test),columns=['Y'])

pred =[]
for i in range(len(test_data)):
	inp = test_data.iloc[i]
	pred.append(prediction(tree,inp))


conf_mat = confusion_matrix(list(test_data.iloc[:,-1]),pred,labels=[0,1])
print("Test accuracy ",(np.sum(np.diagonal(conf_mat))/np.sum(conf_mat))*100,"%")
print(conf_mat)


