from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import sys


# test_f="poker-hand-testing.data"
# train_f="poker-hand-training-true.data"
file = sys.argv[1]
save_as="OneHot_"+ file
with open(file) as f:
	df = pd.read_csv(f,header=None)


# labels = np.array(df.iloc[:,-1]).reshape(len(df),1)
# features = df.iloc[:,:-1]
features = df
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(features)
# data = np.hstack((onehot_encoded,labels))
data = onehot_encoded
df = pd.DataFrame(data)
print(df)
df.to_csv(save_as,index=False)
