import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from pycaret.classification import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer



data = pd.read_csv('Data for Task 1.csv')

# Define preprocessing for numeric and categorical columns
def preprocess(data):
    # Drop 'id' column
    data = data.drop(['id', 'Unnamed: 32'], axis=1)
    
    # Encode 'diagnosis' column to binary
    le = LabelEncoder()
    data['diagnosis'] = le.fit_transform(data['diagnosis'])
    
    return data

df = preprocess(data)

#visualize the data
def visualize_data(df):
    # Plot diagnosis distribution
    sns.countplot(x='diagnosis', data=df)
    plt.xlabel('Diagnosis')
    plt.ylabel('Count')
    plt.title('Diagnosis Distribution')
    plt.show()

    # Plot distribution of features
    df.hist(bins=20, figsize=(20, 20))
    plt.suptitle('Distribution of Features')
    plt.show()

    # Plot correlation heatmap
    corr = df.corr()
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()

    #plot only the features pairs that have a correlation greater than 0.8
    corr = df.corr()
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr[(corr >= 0.8) | (corr <= -0.8)], annot=True, cmap='coolwarm')
    plt.show()

    #print features that are correlated to more than 1 feature with a score of > 0.8
    correlated_features = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.8:
                colname = corr.columns[i]
                correlated_features.add(colname)

    return df 

df2 = visualize_data(df)




    