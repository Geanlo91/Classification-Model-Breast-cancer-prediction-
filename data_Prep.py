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
import dtale 


data = pd.read_csv('Data for Task 1.csv')

#Fuction to understand and visualize the data before preprocessing


# Improved data visualization
def data_visualization(data):
    # Count plot for diagnosis
    plt.figure(figsize=(10, 6))
    sns.countplot(data['diagnosis'])
    plt.title('Diagnosis Count')
    plt.xlabel('Count')
    plt.ylabel('Diagnosis')
    plt.show()

    # Pair plot for features colored by diagnosis
    sns.pairplot(data, hue='diagnosis')
    plt.suptitle('Pair Plot of Features', y=1.02)
    plt.show()
          
    # Heatmap for correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()

    # Box plot for outlier distribution
    plt.figure(figsize=(15, 10))
    data.boxplot()
    plt.title('Outlier Distribution')
    plt.xticks(rotation=90)
    plt.show()

# Example usage
data_visualization(data)
