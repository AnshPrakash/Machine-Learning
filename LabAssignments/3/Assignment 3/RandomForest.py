# Questiom 1 e

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import tree
from pprint import pprint
from functools import reduce



test_f="processesd_credit-cards.test.csv"
validation_f = "processesd_credit-cards.val.csv"
train_f = "processesd_credit-cards.train.csv"
with open(train_f) as f:
	df = pd.read_csv(f)
	df = df.iloc[1:,:]
with open(validation_f) as f:
	df_v = pd.read_csv(f)
	df_v = df_v.iloc[1:,:]
with open(test_f) as f:
	df_t = pd.read_csv(f)
	df_t = df_t.iloc[1:,:]



df_final =pd.DataFrame()
df_final = df_final.append(df)
df_final = df_final.append(df_v)
df_final = df_final.append(df_t)



df_new = pd.get_dummies(df_final, columns=['X3','X4','X6','X7','X8','X9','X10','X11']
						, prefix = ['X3','X4','X6','X7','X8','X9','X10','X11'])


df   = df_new.iloc[:len(df),:]
df_v = df_new.iloc[len(df):len(df)+len(df_v),:]
df_t = df_new.iloc[len(df)+len(df_v):len(df)+len(df_v)+len(df_t),:]



X_train = df.loc[:, df.columns != 'Y' ]
X_train = X_train.iloc[:,1:]
Y_train = df['Y']



X_val = df_v.loc[:, df_v.columns != 'Y' ]
X_val = X_val.iloc[:,1:]
Y_val = df_v['Y']



X_test = df_t.loc[:, df_t.columns != 'Y' ]
X_test = X_test.iloc[:,1:]
Y_test = df_t['Y']




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


pred = clf.predict(X_train)
pred = list(map(int,pred))
Y_train = list(map(int,Y_train))
conf_mat = confusion_matrix(Y_train,pred,labels=[0,1])
accuracy = (np.sum(np.diagonal(conf_mat))/np.sum(conf_mat))*100
print("Accuracy on Training Set by the best Model is ",accuracy,"%")


print(conf_mat)


