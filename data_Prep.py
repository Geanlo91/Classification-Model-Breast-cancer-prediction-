import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from pycaret.classification import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('Data for Task 1.csv')

#Data Exploration
print('Data head:\n', data.head())
print('Data info:\n', data.info())
print('Data shape:\n', data.shape)
print('Data description:\n', data.describe())
#print count of the diagnosis column
print('Diagnosis count:\n', data['diagnosis'].value_counts())

# Define preprocessing for numeric and categorical columns
def preprocess(data):
    # Drop 'id' column
    data = data.drop('id', axis=1)

    #drop columns with all missing or nan values
    data.dropna(axis=1, how='all', inplace=True)
    
    # Encode 'diagnosis' column
    try:
        # Create an instance of LabelEncoder
        le = LabelEncoder()
        # Fit and transform the 'diagnosis' column
        data['diagnosis'] = le.fit_transform(data['diagnosis'])
        # If successful, print success message and assigned values per class
        print("Label encoding was successful.")
        class_mapping = {index: label for index, label in enumerate(le.classes_)}
        print("Assigned values per class:", class_mapping)
    except Exception as e:
        # If an error occurs, print an error message
        print("An error occurred during label encoding:", str(e))
    
    return data
df = preprocess(data)

#visualize the data
def visualize_data(df):
    # Plot diagnosis distribution
    sns.countplot(x='diagnosis', data=df)
    plt.xlabel('Diagnosis')
    plt.ylabel('Count')
    plt.title('Diagnosis Distribution')
    # Set x-ticks labels to 0 = Benign and 1 = Malignant
    plt.xticks(ticks=[0, 1], labels=['Benign', 'Malignant'])
    for i in range(2):
        count = df['diagnosis'].value_counts().values[i]
        plt.text(i, count, str(count), ha='center', va='bottom')
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

    #Boxplot of features to identify outliers
    df.plot(kind='box', subplots=True, layout=(6, 6), figsize=(20, 20))
    plt.suptitle('Boxplot of Features')
    plt.show()

    return df

clean_data = visualize_data(df)


#save the preprocessed data
print(clean_data.head())
clean_data.to_csv('preprocessed_data.csv', index=False)





    