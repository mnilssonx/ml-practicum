import pandas as pd
#from pandas.plotting import scatter_matrix
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv("diabetes.csv")

# 1
#print(data.head())

# 2
#print(data.describe())

#3
#data.hist(); plt.show()
#scatter_matrix(data, alpha=0.2, figsize=(6,6), diagonal='kde'); plt.show()

#4
train, test = train_test_split(data, train_size=0.8)
train, holdout = train_test_split(train, train_size=0.8)
train.to_csv("train.csv")
test.to_csv("test.csv")
holdout.to_csv("holdout.csv")