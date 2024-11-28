from classification import X_test, y_test, X_train, y_train, f1_scores, cv_scores, accuracy_scores
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns
import numpy as np

#Load saved models
baseline = joblib.load('baseline_model.joblib')
decision_tree = joblib.load('decision_tree.joblib')
logistic_regression = joblib.load('logistic_regression.joblib')
rfe_logistic_regression = joblib.load('rfe_logistic_regression.joblib')
random_forest = joblib.load('random_forest_model.joblib')
knn = joblib.load('knn_model.joblib')


#Confusion matrix for each model
def compute_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    return cm

baseline_cm = compute_confusion_matrix(baseline, X_test, y_test)
decision_tree_cm = compute_confusion_matrix(decision_tree, X_test, y_test)
logistic_regression_cm = compute_confusion_matrix(logistic_regression, X_test, y_test)
rfe_logistic_regression_cm = compute_confusion_matrix(rfe_logistic_regression, X_test, y_test)
random_forest_cm = compute_confusion_matrix(random_forest, X_test, y_test)
knn_cm = compute_confusion_matrix(knn, X_test, y_test)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))

confusion_matrices = [
          (baseline_cm, 'Baseline Confusion Matrix'),
          (decision_tree_cm, 'Decision Tree Confusion Matrix'),
          (logistic_regression_cm, 'Logistic Regression Confusion Matrix'),
          (rfe_logistic_regression_cm, 'RFE Logistic Regression Confusion Matrix'),
          (random_forest_cm, 'Random Forest Confusion Matrix'),
          (knn_cm, 'KNN Confusion Matrix')
]

for ax, (cm, title) in zip(axes.flatten(), confusion_matrices):
          disp = ConfusionMatrixDisplay(confusion_matrix=cm)
          disp.plot(ax=ax, cmap='viridis')
          ax.set_title(title)

plt.tight_layout()
plt.show()



#Classification report for each model
def compute_classification_report(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cr = classification_report(y_test, y_pred)
    return cr

model_cr ={'Baseline_cr': compute_classification_report(baseline, X_test, y_test),
          'Decision_tree_cr': compute_classification_report(decision_tree, X_test, y_test),
          'Logistic_regression_cr': compute_classification_report(logistic_regression, X_test, y_test),
          'RFE_Logistic_regression_cr': compute_classification_report(rfe_logistic_regression, X_test, y_test),
          'Random_forest_cr': compute_classification_report(random_forest, X_test, y_test),
          'KNN_cr': compute_classification_report(knn, X_test, y_test)}

for model, cr in model_cr.items():
          print(model)
          print(cr)
          print('\n')


"""Comparing the models based on mean cross-validation scores, mean f1 scores and mean accuracy scores"""
#print mean cross-validation scores for all models in a table with 2 columns: model and mean cross-validation score
cv_scores_df = pd.DataFrame(list(cv_scores.items()), columns=['Model', 'Mean Cross-Validation Score'])
print(cv_scores_df)

#print mean f1 scores for all models in a table with 2 columns: model and mean f1 score
f1_scores_df = pd.DataFrame(list(f1_scores.items()), columns=['Model', 'Mean F1 Score'])
print(f1_scores_df)

#print mean accuracy scores for all models in a table with 2 columns: model and mean accuracy score
accuracy_scores_df = pd.DataFrame(list(accuracy_scores.items()), columns=['Model', 'Mean Accuracy Score'])
print(accuracy_scores_df)


"""Investigating specific features where errors are frequently made"""
# Scatter plots to visualize the distribution of the features for the misclassified samples for all models.
# This helps in identifying patterns or specific features that contribute to the misclassification,
# allowing for better understanding and potential improvement of the model.
fig, scatter_axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 20))

models = {
                      'Baseline': baseline,
                      'Decision Tree': decision_tree,
                      'Logistic Regression': logistic_regression,
                      'RFE Logistic Regression': rfe_logistic_regression,
                      'Random Forest': random_forest,
                      'KNN': knn
}
for ax, (model_name, model) in zip(scatter_axes.flatten(), models.items()):
                        y_pred = model.predict(X_test)
                        errors = X_test[y_test != y_pred]
                        
                        for feature in X_test.columns:
                            ax.scatter(errors[feature], y_test[errors.index], color='red', label='Misclassified', alpha=0.2)
                        
                        ax.set_title(f'{model_name} Errors')
                        ax.set_xlabel('Features')
                        ax.set_ylabel('True Labels')
                        ax.legend(['Misclassified'])

plt.tight_layout()
plt.show()

#identify the column name and data points with the misclassified samples
for model_name, model in models.items():
    y_pred = model.predict(X_test)
    errors = X_test[y_test != y_pred]
    print(f'Misclassified samples for {model_name}:')
    print(errors)
    print('\n')
    
    """Print only the common columns"""