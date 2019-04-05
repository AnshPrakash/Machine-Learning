import pandas as pd
import numpy as np
from math import log
from pprint import pprint
import sys



def selectBestAttribute(df,attributes):
	'''
		select the best attribute to split from the data 
		return a sting from attributes available
	'''
	return(attribute[np.argmax(list(map(getInformationGain,attribute)))])
	

def split(df,attribute):
	'''
		Based on the given attribute split the data 
		and return the data frames
	'''
	pass

def classify(labels):
	'''
		returns the classified label
	'''
	pass

def getInformationGain(data,attribute):
	'''
		returns the information gain by splitiing  
		the data on the given attribute
	'''
	total = len(data)
	Py1 = sum(data['Y']==1)/total
	#  py1 == 1 or 0 we don't need to calculate this
	hy = -(Py1*log(Py1,2) + (1-Py1)*log((1-Py1),2) )
	hy_x = 0.0
	for val in values_set[attribute]:
		# print(data[attribute] == values_set[attribute][val])
		df_temp = data[data[attribute] == val]
		px = len(df_temp)/total
		if px != 0:
			py1_x = sum(df_temp['Y'] == 1)/len(df_temp)
			if py1_x == 0 or py1_x==1:
				pass
			else:
				hy_x -= px*(py1_x*log(py1_x,2) + (1-py1_x)*log((1-py1_x),2))

	informationGain = hy -hy_x
	return(informationGain)


def checkDataPure(labels):
	'''	
		check if the data is pure or not
	'''
	return(False if len(set(labels)) > 1 else True )


def buildDecisionTree(data,labels):
	'''
		
		returns a Dictionary(which actually is a tree)
	'''
	pass




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
values_set = {}
del df 
for attribute in attributes:
	values_set[attribute] =list(set(data[attribute]))
tree = buildDecisionTree(data,labels,values_set)



for attribute in attributes:
	print(attribute,getInformationGain(data,attribute))

	
