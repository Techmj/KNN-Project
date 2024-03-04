print('hi')
dadslk

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix

dataLungCancer = pd.read_csv('survey lung cancer.csv')
print(dataLungCancer.shape)
print(dataLungCancer.head())

# Print the summary of the data
print(dataLungCancer.describe())
print(dataLungCancer.describe().T)

# Print the datatypes which are keys.
types = dataLungCancer.dtypes
print(types)
print("Keys of dataLungCancer dataset:\n", dataLungCancer.keys())

# print the target variable
print("Mood:", dataLungCancer['mood'])
