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
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import joblib



# Load data
data = pd.read_csv('preprocessed_data.csv')
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionaries to store scores
cv_scores = {}
f1_scores = {}
accuracy_scores = {}

def plot_feature_importances(importances, title):
    """Plot feature importances."""
    features = [f'PC{i+1}' for i in range(len(importances))]
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
    plt.xlabel('Principal Components')
    plt.ylabel('Importance')
    plt.title(title)
    plt.show()

def baseline_model(X_train, X_test, y_train, y_test):
    """Train and evaluate a baseline model."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)

    #save the trained model
    joblib.dump(dummy, 'baseline_model.joblib')
    print("Model saved as 'baseline_model.joblib'.")

    y_pred = dummy.predict(X_test)

    f1 = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
    f1_scores['Baseline'] = f1
    
    scores = cross_val_score(dummy, X_train, y_train, cv=5)
    mean_score = np.mean(scores)
    cv_scores['Baseline'] = mean_score
    print(f'Baseline cross-validation scores: {scores}')

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores['Baseline'] = accuracy

baseline_model(X_train, X_test, y_train, y_test)


def decision_tree(X_train, X_test, y_train, y_test):
    """Train and evaluate a decision tree model."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(max_depth=5, random_state=42))
    ])
    pipe.fit(X_train, y_train)

    #save the trained model
    joblib.dump(pipe, 'decision_tree.joblib')
    print("Model saved as 'decision_tree.joblib'.")

    y_pred = pipe.predict(X_test)

    f1 = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
    f1_scores['Decision Tree'] = f1
    
    scores = cross_val_score(pipe, X_train, y_train, cv=5)
    mean_score = np.mean(scores)
    cv_scores['Decision Tree'] = mean_score
    print(f'Decision tree cross-validation scores: {scores}')

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores['Decision Tree'] = accuracy

    importances = pipe.named_steps['classifier'].feature_importances_
    plot_feature_importances(importances, 'Decision Tree Feature Importances')

decision_tree(X_train, X_test, y_train, y_test)


def logistic_regression(X_train, X_test, y_train, y_test):
    """Train and evaluate a logistic regression model."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipe.fit(X_train, y_train)

    #save the trained model
    joblib.dump(pipe, 'logistic_regression.joblib')
    print("Model saved as 'logistic_regression.joblib'.")

    y_pred = pipe.predict(X_test)

    f1 = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
    f1_scores['Logistic Regression'] = f1
    
    scores = cross_val_score(pipe, X_train, y_train, cv=5)
    print(f'Logistic regression cross-validation scores: {scores}')
    mean_score = np.mean(scores)
    cv_scores['Logistic Regression'] = mean_score

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores['Logistic Regression'] = accuracy

    importances = np.abs(pipe.named_steps['classifier'].coef_[0])
    plot_feature_importances(importances, 'Logistic Regression Feature Importances')

logistic_regression(X_train, X_test, y_train, y_test)

def rfe_logistic_regression(X_train, X_test, y_train, y_test):
    """Train and evaluate a logistic regression model with recursive feature elimination."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rfe', RFE(LogisticRegression(max_iter=100, random_state=42))),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))])

    pipe.fit(X_train, y_train)

    #save the trained model
    joblib.dump(pipe, 'rfe_logistic_regression.joblib')
    print("Model saved as 'rfe_logistic_regression.joblib'.")

    y_pred = pipe.predict(X_test)

    f1 = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
    f1_scores['RFE Logistic Regression'] = f1
    
    scores = cross_val_score(pipe, X_train, y_train, cv=5)
    print(f'RFE logistic regression cross-validation scores: {scores}')
    mean_score = np.mean(scores)
    cv_scores['RFE Logistic Regression'] = mean_score

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores['RFE Logistic Regression'] = accuracy

    importances = pipe.named_steps['rfe'].ranking_ / np.max(pipe.named_steps['rfe'].ranking_)
    plot_feature_importances(importances, 'RFE Logistic Regression Feature Importances')

rfe_logistic_regression(X_train, X_test, y_train, y_test)


def knn(X_train, X_test, y_train, y_test):
    """Train and evaluate a k-nearest neighbors model."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier(n_neighbors=10))
    ])
    pipe.fit(X_train, y_train)

    #save the trained model
    joblib.dump(pipe, 'knn_model.joblib')
    print("Model saved as 'knn_model.joblib'.")

    y_pred = pipe.predict(X_test)

    f1 = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
    f1_scores['KNN'] = f1
    
    scores = cross_val_score(pipe, X_train, y_train, cv=5)
    print(f'KNN cross-validation scores: {scores}')
    mean_score = np.mean(scores)
    cv_scores['KNN'] = mean_score

knn(X_train, X_test, y_train, y_test)


def random_forest(X_train, X_test, y_train, y_test):
    """Train and evaluate a random forest model."""
    pipe = Pipeline([
        ('scaler', QuantileTransformer()),
        ('pca', PCA(n_components=10)),
        ('classifier', RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42))
    ])
    pipe.fit(X_train, y_train)

    #save the trained model
    joblib.dump(pipe, 'random_forest_model.joblib')
    print("Model saved as 'random_forest_model.joblib'.")

    y_pred = pipe.predict(X_test)

    f1 = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
    f1_scores['Random Forest'] = f1
    
    scores = cross_val_score(pipe, X_train, y_train, cv=5)
    print(f'Random forest cross-validation scores: {scores}')
    mean_score = np.mean(scores)
    cv_scores['Random Forest'] = mean_score

    importances = pipe.named_steps['classifier'].feature_importances_
    plot_feature_importances(importances, 'Random Forest Feature Importances')

random_forest(X_train, X_test, y_train, y_test)













