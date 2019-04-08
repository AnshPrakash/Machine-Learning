import pandas as pd
import numpy as np
from math import log,isnan
from pprint import pprint
from itertools import repeat
import operator
from functools import reduce
from sklearn.metrics import confusion_matrix
import sys

import matplotlib.pyplot as plt

Depth_count ={}
values_set = {}
data = []
# num_ofnodes = 0
def selectBestAttribute(indices,attributes):
	'''
		select the best attribute to split from the data 
		return a sting from attributes available
	'''
	data_tp = data.iloc[indices,:]
	total = len(data_tp)
	Py1 = np.mean(data_tp['Y'])
	ig  = np.array(list(map(getInformationGain,repeat(data_tp),attributes,repeat(Py1))))
	idx = np.argmax(ig)
	return(attributes[idx],ig[idx])
	

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
	y_entropy       = -(Py1*(np.log2(Py1)) + (1-Py1)*np.log2(1-Py1))
	children        = list(map(lambda val: data_tp[data_tp[attribute]==val].iloc[:,-1] ,values_set[attribute]))
	px_children     = list(map(lambda df:len(df)/len(data_tp),children))
	cond_py1x       = list(map(lambda lb: np.mean(lb),children))
	entopies        = list(map(lambda p: 0 if (p==0 or p==1 or isnan(p))  else -p*np.log2(p)-(1-p)*np.log2(1-p),cond_py1x))
	avg_ent         = list(map(operator.mul,entopies,px_children))
	hy_x            = reduce(operator.add,avg_ent)
	informationGain = y_entropy - hy_x
	return(informationGain)


def checkDataPure(indices):
	'''	
		check if the data is pure or not
	'''
	if len(indices)>0:
		labels = data.iloc[indices,-1]
		return(np.all(labels  == labels[indices[0]]))
	else:
		return(True)



def prediction(tree,input):
	if input[tree[0]] not in values_set[tree[0]]:
		return(1 if data.iloc[tree[2],-1].mean() >0.5 else 0)
	if isinstance(tree[1][input[tree[0]]],int):
		return(tree[1][input[tree[0]]])
	else:
		return(prediction(tree[1][input[tree[0]]],input))

def specialPrediction(tree,input,n_nodes,depth):
	'''
		for Ploting Node vs Accuracy Graph
	'''
	if input[tree[0]] not in values_set[tree[0]]:
		return(1 if data.iloc[tree[2],-1].mean() >0.5 else 0)
	if isinstance(tree[1][input[tree[0]]],int):
		return(tree[1][input[tree[0]]])
	else:
		if Depth_count[depth] < n_nodes:
			return(1 if data.iloc[tree[2],-1].mean() >0.5 else 0)
		else:
			return(specialPrediction(tree[1][input[tree[0]]],input,n_nodes,depth+1))

	

def countDepthNode(depth,tree):
	if depth not in Depth_count:
		Depth_count[depth] = 1
	else:
		Depth_count[depth] += 1 
	for val in tree[1]:
		if isinstance(tree[1][val],int):
			if depth+1 not in Depth_count:
				Depth_count[depth+1] = 1
			else:
				Depth_count[depth+1] += 1 
		else:
			countDepthNode(depth+1,tree[1][val])





def buildDecisionTree(indices,attributes):
	'''
		
		returns a Dictionary(which actually is a tree)
	'''
	tree = {}
	# global num_ofnodes
	# num_ofnodes += 1
	if len(indices)<=10 or len(attributes)==1 or checkDataPure(indices) :
		return(classify(indices))
	attribute,ig = selectBestAttribute(indices,attributes)
	new_attributes = [] 
	for att in attributes:
		if att != attribute:
			new_attributes.append(att)
	if ig == 0:
		return(classify(indices))
	splited_data =split(attribute,indices)
	temp = {}
	for val in splited_data:
		if len(splited_data[val]) == 0:
			temp[val] = classify(indices) #No occueence of this value in the data given
		else:
			temp[val] = buildDecisionTree(splited_data[val],new_attributes)
	tree = [attribute,temp,indices]
	return(tree)


def getAllNodePredictions(test,max_depth):
	l = []
	for d in range(max_depth+1):
		pred =[]
		for i in range(len(test)):
			inp = test.iloc[i]
			pred.append(specialPrediction(tree,inp,Depth_count[d],0))
		conf_mat = confusion_matrix(list(test.iloc[:,-1]),pred,labels=[0,1])
		print(d,(np.sum(np.diagonal(conf_mat))/np.sum(conf_mat))*100)
		l.append((np.sum(np.diagonal(conf_mat))/np.sum(conf_mat))*100)
	return(l)



validation_f = "processesd_credit-cards.val.csv"
test_f = "processesd_credit-cards.test.csv"
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

tree = buildDecisionTree(data.index[[True]*len(data)],attributes)
print(countDepthNode(0,tree),"Count function")
total_nodes = 0
max_depth = -1
td_count = {}
for i in Depth_count:
	td_count[i] = Depth_count[i]
	total_nodes += Depth_count[i]
	max_depth+=1

print("Total Num of Nodes",total_nodes)
print("Max Depth ",max_depth)

for d in range(1,max_depth+1):
	Depth_count[d] += Depth_count[d-1]



pred =[]
for i in range(len(data)):
	inp = data.iloc[i]
	pred.append(prediction(tree,inp))


conf_mat = confusion_matrix(list(data.iloc[:,-1]),pred,labels=[0,1])
print("accuracy ",(np.sum(np.diagonal(conf_mat))/np.sum(conf_mat))*100,"%")
print(conf_mat)
pprint(values_set)


with open(test_f) as f:
	df = pd.read_csv(f)

test_data = (df.iloc[1:,1:]) ## contains labels and data both (don't need X0)
labels_test = df.iloc[:,-1]
pred =[]
for i in range(len(test_data)):
	inp = test_data.iloc[i]
	pred.append(prediction(tree,inp))


conf_mat = confusion_matrix(list(test_data.iloc[:,-1]),pred,labels=[0,1])
print("Test accuracy ",(np.sum(np.diagonal(conf_mat))/np.sum(conf_mat))*100,"%")
print(conf_mat)


with open(validation_f) as f:
	df = pd.read_csv(f)

val_data = (df.iloc[1:,1:]) ## contains labels and data both (don't need X0)
labels_val = df.iloc[:,-1]
pred =[]
for i in range(len(val_data)):
	inp = val_data.iloc[i]
	pred.append(prediction(tree,inp))


conf_mat = confusion_matrix(list(val_data.iloc[:,-1]),pred,labels=[0,1])
print("Validation Set accuracy ",(np.sum(np.diagonal(conf_mat))/np.sum(conf_mat))*100,"%")
print(conf_mat)



val_l = getAllNodePredictions(val_data,max_depth)
temp_l=[]
for d  in range(max_depth+1):
	temp_l+=[val_l[d]]*td_count[d]

val_l = temp_l
test_l = getAllNodePredictions(test_data,max_depth)
temp_l=[]
for d  in range(max_depth+1):
	temp_l+=[test_l[d]]*td_count[d]
test_l = temp_l
train_l = getAllNodePredictions(data,max_depth)
temp_l=[]
for d  in range(max_depth):
	temp_l+=[train_l[d]]*td_count[d]
train_l = temp_l

plt.style.use("ggplot")
plt.figure()
plt.plot(list(range(1,len(val_l)+1)),val_l,label="Validation accuracy")
plt.plot(list(range(1,len(test_l)+1)),test_l,label = "Test accuracy")
plt.plot(list(range(1,len(train_l)+1)),train_l,label = "Training accuracy")
plt.ylim(0, 100)


plt.title('Accuracy Vs Num Of Nodes')
plt.xlabel("# Nodes")
plt.ylabel("Accuracy %")
plt.legend(loc="upper right")
plt.show()


