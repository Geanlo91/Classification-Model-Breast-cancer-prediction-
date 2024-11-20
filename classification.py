import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

data = pd.read_csv('Data for Task 1.csv')
print(data.head())

X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
