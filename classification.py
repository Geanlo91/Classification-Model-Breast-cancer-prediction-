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
from sklearn.feature_selection import RFE



data = pd.read_csv('preprocessed_data.csv')

X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with three steps: scaling, PCA, and classification
pipe = Pipeline([('scaler', QuantileTransformer()),  # Scale features to a uniform distribution
                 ('pca', PCA(n_components=20)),  # Reduce dimensionality to 20 principal components
                 ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])  # Train a random forest classifier
# Fit the pipeline to the training data
pipe.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = pipe.predict(X_test)

# Print the classification report to evaluate the model's performance
print(classification_report(y_test, y_pred))

# Compute the cross-validation scores of the pipeline
scores = cross_val_score(pipe, X_train, y_train, cv=5)
print(f'Cross-validation scores: {scores}')
print(f'Mean cross-validation score: {np.mean(scores)}')

# Plot the feature importances with feature names
importances = pipe.named_steps['classifier'].feature_importances_
features = [f'PC{i+1}' for i in range(len(importances))]
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
plt.xlabel('Principal Components')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()


# Get the original feature names from the PCA transformation
pca = pipe.named_steps['pca']
most_important_features = np.abs(pca.components_).sum(axis=0).argsort()[::-1][:20]
original_features = X.columns[most_important_features]

# Match the most important features to the original feature names based on the PCA components
important_features_indices = np.argsort(importances)[::-1][:20]
important_original_features = [original_features[i] for i in range(len(original_features))]

# Plot the most important original features
plt.figure(figsize=(10, 6))
plt.bar(range(len(important_original_features)), importances[important_features_indices])
plt.xticks(range(len(important_original_features)), important_original_features, rotation=90)
plt.xlabel('Original Features')
plt.ylabel('Importance')
plt.title('Most Important Original Features')
plt.show()










