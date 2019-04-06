# Questiom 1 d

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import tree
from pprint import pprint
from sklearn.preprocessing import OneHotEncoder


test_f="processesd_credit-cards.test.csv"
validation_f = "processesd_credit-cards.val.csv"
train_f = "processesd_credit-cards.train.csv"
with open(train_f) as f:
	df = pd.read_csv(f)
with open(validation_f) as f:
	df_v = pd.read_csv(f)
with open(test_f) as f:
	df_t = pd.read_csv(f)


X_train = (df.iloc[1:,2:-1])
Y_train = df.iloc[1:,-1]


X_val = (df_v.iloc[1:,2:-1]) 
Y_val = df_v.iloc[1:,-1]

X_test = (df_t.iloc[1:,2:-1]) 
Y_test = df_t.iloc[1:,-1]


max_depths = [None,100,80,70,50,10]
min_samples_leafs  = [1,10,50] 
min_samples_splits = [2,10,100]


acc = 0.0
best_model = []
for max_depth in max_depths:
	for min_samples_split in min_samples_splits:
		for min_samples_leaf in min_samples_leafs:
			clf = DecisionTreeClassifier(	criterion = 'entropy', 
											max_depth = max_depth,
											min_samples_leaf = min_samples_leaf,
											min_samples_split = min_samples_split,
										)
			clf.fit(X_train, Y_train)
			pred = clf.predict(X_val)
			pred = list(map(int,pred))
			Y_val = list(map(int,Y_val))
			conf_mat = confusion_matrix(Y_val,pred,labels=[0,1])
			accuracy = (np.sum(np.diagonal(conf_mat))/np.sum(conf_mat))*100
			(best_model,acc) = (clf,accuracy) if accuracy>acc else (best_model,acc)
			print( "max_depth ,min_sample_split,min_sample_leaf:", max_depth,min_samples_split,min_samples_leaf)
			print(" accuracy :",accuracy,"%")


print("\n\nBest Model with best accuracy on validation set")
print(best_model," ",acc,"%")


clf = best_model
pred = clf.predict(X_test)
pred = list(map(int,pred))
Y_test = list(map(int,Y_test))
conf_mat = confusion_matrix(Y_test,pred,labels=[0,1])
accuracy = (np.sum(np.diagonal(conf_mat))/np.sum(conf_mat))*100
print("Accuracy on Test Set by the best Model is ",accuracy,"%")



print(conf_mat)



