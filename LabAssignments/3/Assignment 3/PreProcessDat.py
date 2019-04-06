import pandas as pd
import numpy as np
import sys



test_f="credit-cards.test.csv"
validation_f = "credit-cards.val.csv"
train_f = "credit-cards.train.csv"

def function(file):
	with open(file) as f:
		df = pd.read_csv(f)


	data = (df.iloc[1:,:])
	# mat =np.array(data)
	# print(mat)
	attributes = list(df.columns.values)

	# print(data)
	# print(attributes)
	# continous data
	med_data =data.median(axis = 0)
	# print(med_data)

	for i in [1,5]+list(range(12,24)):
		data[attributes[i]] = (data[attributes[i]].astype(int) >med_data[attributes[i]]).astype(int)


	data.to_csv("processesd_"+file)


function(train_f)
function(test_f)
function(validation_f)