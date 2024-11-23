import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif


data = pd.read_csv('preprocessed_data.csv')

X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with three steps: scaling, PCA, and classification
pipe = Pipeline([('scaler', QuantileTransformer()),  # Scale features to a uniform distribution
                 ('pca', PCA(n_components=5)),  # Reduce dimensionality to 5 principal components
                 ('classifier', RandomForestClassifier(max_depth=20, n_estimators=100, min_samples_split=5))])  # Random Forest classifier
# Fit the pipeline to the training data
pipe.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = pipe.predict(X_test)

# Print the classification report to evaluate the model's performance
print(classification_report(y_test, y_pred))



scores = cross_val_score(pipe, X_train, y_train, cv=5)
print(f'Cross-validation scores: {scores}')
print(f'Mean cross-validation score: {np.mean(scores)}')


#add RFE Feature selection 






