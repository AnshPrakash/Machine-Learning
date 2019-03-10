# python3
import time 
import numpy as np
import pandas as pd
from cvxopt import matrix
from cvxopt import solvers
from sklearn.metrics import confusion_matrix

start_time=time.time()
# get the support vector
entry_no=7
start_time=time.time()
with open('./mnist/train.csv') as f:
	df = pd.read_csv(f,header=None)

df1=(df.loc[(df.iloc[:,-1]==entry_no) | (df.iloc[:,-1]==(entry_no+1)%10)])

df1.iloc[:,-1]=df1.iloc[:,-1].replace({entry_no:1,(entry_no+1)%10:-1})
df1.iloc[:,:-1]=df1.iloc[:,:-1]/255
