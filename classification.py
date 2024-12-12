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
import joblib
from sklearn.model_selection import GridSearchCV
import shap


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

def plot_feature_importances(importances, title, feature_names=None):
    """Plot feature importances."""
    if feature_names is None:
        feature_names = [f'PC{i+1}' for i in range(len(importances))]
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
    plt.xlabel('Features')
    plt.show()

def baseline_model(X_train, X_test, y_train, y_test):
    """Train and evaluate a baseline model."""
    dummy = DummyClassifier(strategy="uniform", random_state=42)
    dummy.fit(X_train, y_train)

    #save the trained model
    joblib.dump(dummy, 'baseline_model.joblib')
    print("Model saved as 'baseline_model.joblib'.")

    y_pred = dummy.predict(X_test)

    f1 = f1_score(y_test, y_pred, average='weighted')
    f1_scores['Baseline'] = f1
    
    scores = cross_val_score(dummy, X_train, y_train, cv=5)
    mean_score = np.mean(scores)
    cv_scores['Baseline'] = mean_score

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores['Baseline'] = accuracy

baseline_model(X_train, X_test, y_train, y_test)


def decision_tree(X_train, X_test, y_train, y_test):
    """Train and evaluate a decision tree model with grid search."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(random_state=42))])
    
    param_grid = {
        'classifier__max_depth': [3, 5, 10, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]}

    grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(grid_search, 'decision_tree.joblib')
    print("Model saved as 'decision_tree.joblib'.")

    y_pred = grid_search.predict(X_test)

    f1 = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
    f1_scores['Decision Tree'] = f1
    
    scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5)
    mean_score = np.mean(scores)
    cv_scores['Decision Tree'] = mean_score

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores['Decision Tree'] = accuracy

    #Feature importance
    importances = grid_search.best_estimator_.named_steps['classifier'].feature_importances_

    # Mapping feature importances to original feature names
    feature_names = X_train.columns
    plot_feature_importances(importances, 'Decision Tree Feature Importances', feature_names)
    
decision_tree(X_train, X_test, y_train, y_test)


import shap

def logistic_regression(X_train, X_test, y_train, y_test):
    """Train and evaluate a logistic regression model with grid search."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=500, random_state=42))])
    
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear', 'lbfgs']}

    grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(grid_search, 'logistic_regression.joblib')
    print("Model saved as 'logistic_regression.joblib'.")

    y_pred = grid_search.predict(X_test)

    f1 = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
    f1_scores['Logistic Regression'] = f1
    
    scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5)
    mean_score = np.mean(scores)
    cv_scores['Logistic Regression'] = mean_score

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores['Logistic Regression'] = accuracy

    #Feature importance
    importances = np.abs(grid_search.best_estimator_.named_steps['classifier'].coef_[0])

    # Mapping feature importances to original feature names
    feature_names = X_train.columns
    plot_feature_importances(importances, 'Logistic Regression Feature Importances', feature_names)

logistic_regression(X_train, X_test, y_train, y_test)



def rfe_logistic_regression(X_train, X_test, y_train, y_test):
    #Train and evaluate a logistic regression model with RFE and grid search.
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rfe', RFE(LogisticRegression(max_iter=100, random_state=42))),
        ('classifier', LogisticRegression(max_iter=500, random_state=42))])
    
    param_grid = {
        'rfe__n_features_to_select': [5, 10, 15],
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear', 'lbfgs']}
    
    grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(grid_search, 'rfe_logistic_regression.joblib')
    print("Model saved as 'rfe_logistic_regression.joblib'.")

    y_pred = grid_search.predict(X_test)

    f1 = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
    f1_scores['RFE Logistic Regression'] = f1
    
    scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5)
    mean_score = np.mean(scores)
    cv_scores['RFE Logistic Regression'] = mean_score

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores['RFE Logistic Regression'] = accuracy

    #Feature importance
    importances = np.abs(grid_search.best_estimator_.named_steps['classifier'].coef_[0])

    # Mapping feature importances to original feature names
    rfe = grid_search.best_estimator_.named_steps['rfe']
    feature_names = X_train.columns[rfe.support_]
    plot_feature_importances(importances, 'RFE Logistic Regression Feature Importances', feature_names)

rfe_logistic_regression(X_train, X_test, y_train, y_test)


def knn(X_train, X_test, y_train, y_test):
    """Train and evaluate a k-nearest neighbors model."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier())])

    param_grid = {
        'classifier__n_neighbors': [3, 5, 10, 20],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__metric': ['euclidean', 'manhattan', 'minkowski']}
    
    grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    #save the trained model
    joblib.dump(grid_search, 'knn_model.joblib')
    print("Model saved as 'knn_model.joblib'.")

    y_pred = grid_search.predict(X_test)

    f1 = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
    f1_scores['KNN'] = f1
    
    scores = cross_val_score(pipe, X_train, y_train, cv=5)
    mean_score = np.mean(scores)
    cv_scores['KNN'] = mean_score

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores['KNN'] = accuracy

knn(X_train, X_test, y_train, y_test)


def random_forest_with_grid_search(X_train, X_test, y_train, y_test):
    """Train and evaluate a random forest model with Grid Search."""
    pipe = Pipeline([
        ('scaler', QuantileTransformer()),
        ('pca', PCA(n_components=18)),
        ('classifier', RandomForestClassifier(random_state=42))])

    # Define hyperparameter grid
    param_grid = {
        'classifier__n_estimators': [100, 200, 500],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__max_features': ['sqrt', 'log2', None],}

    grid_search = GridSearchCV(pipe, param_grid, cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Save the best model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'random_forest_model.joblib')
    print("Model saved as 'random_forest_model.joblib'.")
    print(f"Best Parameters: {grid_search.best_params_}")

    # Evaluate the best model
    y_pred = best_model.predict(X_test)
    f1 = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
    f1_scores['Random Forest'] = f1

    scores = cross_val_score(best_model, X_train, y_train, cv=5)
    mean_score= np.mean(scores)
    cv_scores['Random Forest'] = mean_score

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores['Random Forest'] = accuracy

    #Feature importance
    classifier = best_model.named_steps['classifier']
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        feature_names = [f'PC{i+1}' for i in range(importances.shape[0])]
        plot_feature_importances(importances, 'Random Forest Feature Importances (PCA)', feature_names)
    else:
        print("Feature importances not available for the selected model.")
   
  
random_forest_with_grid_search(X_train, X_test, y_train, y_test)

