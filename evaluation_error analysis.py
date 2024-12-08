from classification import X_test, y_test, X_train, f1_scores, cv_scores, accuracy_scores
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
import os

#Load saved models
models = {
    'Baseline': joblib.load('baseline_model.joblib'),
    'Decision Tree': joblib.load('decision_tree.joblib'),
    'Logistic Regression': joblib.load('logistic_regression.joblib'),
    'RFE Logistic Regression': joblib.load('rfe_logistic_regression.joblib'),
    'Random Forest': joblib.load('random_forest_model.joblib'),
    'KNN': joblib.load('knn_model.joblib')}


#Confusion matrix for each model
def compute_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    return cm
#Generate confusion matrices for all models
confusion_matrices = {model_name: compute_confusion_matrix(model, X_test, y_test) for model_name, model in models.items()}
#Plot confusion matrix for all models
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))
for ax, (model_name, cm) in zip(axes.flatten(), confusion_matrices.items()):
          disp = ConfusionMatrixDisplay(confusion_matrix=cm)
          disp.plot(ax=ax, cmap='viridis')
          ax.set_title(f'{model_name} Confusion Matrix')
plt.tight_layout()
plt.show()


#Function for classification report of each model
def compute_classification_report(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cr = classification_report(y_test, y_pred)
    return cr
#Generate classification reports for all models
classification_reports = {model_name: compute_classification_report(model, X_test, y_test) for model_name, model in models.items()}
#Print classification reports for all models
for model, cr in classification_reports.items():
          print(model)
          print(cr)
          print('\n')


#Comparing the models based on mean cross-validation scores, mean f1 scores and mean accuracy scores
cv_scores_df = pd.DataFrame(list(cv_scores.items()), columns=['Model', 'Mean Cross-Validation Score'])
f1_scores_df = pd.DataFrame(list(f1_scores.items()), columns=['Model', 'Mean F1 Score'])
accuracy_scores_df = pd.DataFrame(list(accuracy_scores.items()), columns=['Model', 'Mean Accuracy Score'])

print(cv_scores_df)
print(f1_scores_df)
print(accuracy_scores_df)

#Plotting line graphs in a figure to compare the models based on mean cross-validation scores, mean f1 scores and mean accuracy scores
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))
cv_scores_df.plot(kind='line', x='Model', y='Mean Cross-Validation Score', ax=axes[0], color='blue', legend=False)
f1_scores_df.plot(kind='line', x='Model', y='Mean F1 Score', ax=axes[1], color='red', legend=False)
accuracy_scores_df.plot(kind='line', x='Model', y='Mean Accuracy Score', ax=axes[2], color='green', legend=False)
axes[0].set_title('Mean Cross-Validation Scores')
axes[1].set_title('Mean F1 Scores')
axes[2].set_title('Mean Accuracy Scores')
for ax in axes:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.show()


"""Investigating specific features where errors are frequently made"""
# Scatter plots to visualize the distribution of the features for the misclassified samples for all models.
# This helps in identifying patterns or specific features that contribute to the misclassification,
# allowing for better understanding and potential improvement of the model.
fig, scatter_axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 20))
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
    if model == models['Baseline']:
        continue 
    y_pred = model.predict(X_test)
    errors = X_test[y_test != y_pred]
    print(f'Misclassified samples for {model_name}:')
    print(errors)
    print('\n')

#print only the rows with masclassified samples that are common across all models
common_errors = X_test[
            (y_test != models['Baseline'].predict(X_test)) &
            (y_test != models['Decision Tree'].predict(X_test)) &
            (y_test != models['Logistic Regression'].predict(X_test)) &
            (y_test != models['RFE Logistic Regression'].predict(X_test)) &
            (y_test != models['Random Forest'].predict(X_test)) &
            (y_test != models['KNN'].predict(X_test))
            ]
print('Common misclassified samples across all models:')
print(common_errors)
print('\n')


#PCA for visualization of misclassified samples
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)

misclassified_indices = np.zeros_like(y_test, dtype=bool)
for model in models.values():
    misclassified_indices |= (y_test != model.predict(X_test))
true_labels = y_test[misclassified_indices]

misclassified_rows = X_test[misclassified_indices]
misclassified_rows['True Label'] = true_labels

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[~misclassified_indices, 0], X_pca[~misclassified_indices, 1], 
            c=y_test[~misclassified_indices], cmap='coolwarm', alpha=0.6, label='Correctly Classified')
plt.scatter(X_pca[misclassified_indices, 0], X_pca[misclassified_indices, 1], c=true_labels, cmap='autumn', 
            edgecolor='k', s=100, label='Misclassified')
plt.legend()
plt.title('PCA Visualization of Misclassifications')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


"""Evaluating model interpretability for misclassified samples, Comparing the means
of misclassified and correctly classified samples for each model"""

# Identify misclassified and correctly classified rows for each model
misclassified = []
correctly_classified = []

for model_name, model in models.items():
    y_pred = model.predict(X_test)
    misclassified.append(X_test[y_test != y_pred])
    correctly_classified.append(X_test[y_test == y_pred])

misclassified = pd.concat(misclassified)
correctly_classified = pd.concat(correctly_classified)
misclassified_summary = misclassified.describe().T
correctly_classified_summary = correctly_classified.describe().T
#save the summary statistics in a dataframe
misclassified_df = pd.DataFrame(misclassified_summary)
correctly_classified_df = pd.DataFrame(correctly_classified_summary)
# Align on shared statistics
misclassified_mean = misclassified_df['mean']
correctly_classified_mean = correctly_classified_df['mean']

# Compute and display the differences
mean_diff = misclassified_mean - correctly_classified_mean
diff_df = pd.DataFrame({'Mean Difference': mean_diff, 
                        'Misclassified Mean': misclassified_mean, 
                        'Correctly Classified Mean': correctly_classified_mean})

#plot a line grapH for the features to compare the means of misclassified and correctly classified samples for all models.
plt.figure(figsize=(10, 6))
plt.plot(misclassified_mean, label='Misclassified Mean', marker='o')
plt.plot(correctly_classified_mean, label='Correctly Classified Mean', marker='o')
plt.xlabel('Features')
plt.ylabel('Mean')
plt.title('Mean Comparison of Misclassified and Correctly Classified Samples')
plt.xticks(rotation=90)
plt.legend()
plt.show()


# LIME Explanation for a specific prediction
def lime_explanation(model, X_train, X_test, i):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        mode='classification',
        class_names=['Negative', 'Positive'],
        feature_names=X_train.columns,
        discretize_continuous=True)
    exp = explainer.explain_instance(X_test.iloc[i], model.predict_proba)
    exp.show_in_notebook()
    return lime_explanation
#visualize the explanation for all the models
for model_name, model in models.items():
    if model == models['Baseline']:
        continue  # Skip the baseline model
    print(f'LIME Explanation for {model_name}:')
    lime_explanation(model, X_train, X_test, 20)
    print('\n')

# Function for LIME Explanation for a range of misclassified samples
def lime_explanation_for_errors(model, X_train, X_test, y_test, model_name):
    # Create the LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        mode='classification',
        class_names=['Negative', 'Positive'],
        feature_names=X_train.columns,
        discretize_continuous=True)
    
    # Identify misclassified samples
    y_pred = model.predict(X_test)
    misclassified_indices = np.where(y_pred != y_test)[0]
    
    explanations = {}
    # Apply LIME to each misclassified sample and visualize
    for i in misclassified_indices:
        exp = explainer.explain_instance(X_test.iloc[i], model.predict_proba)
        explanations[i] = exp
        print(f'LIME Explanation for misclassified sample {i} using {model_name}:')
        exp.show_in_notebook()  # This shows the LIME explanation for each misclassified instance
    return explanations



# Function for LIME Explanation for false positives and false negatives
def lime_explanation_for_fp_fn(model, X_train, X_test, y_test, model_name):
    # Create the LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        mode='classification',
        class_names=['Negative', 'Positive'],
        feature_names=X_train.columns,
        discretize_continuous=True)
    
    # Identify false positives and false negatives
    y_pred = model.predict(X_test)
    fp_indices = np.where((y_pred == 1) & (y_test == 0))[0]
    fn_indices = np.where((y_pred == 0) & (y_test == 1))[0]

    # Apply LIME to false positives
    for i in fp_indices:
        exp = explainer.explain_instance(X_test.iloc[i], model.predict_proba)
        print(f'LIME Explanation for False Positive sample {i} using {model_name}:')
        exp.show_in_notebook(show_table=True)  # Shows the LIME explanation for the false positive instance
        
    # Apply LIME to false negatives
    for i in fn_indices:
        exp = explainer.explain_instance(X_test.iloc[i], model.predict_proba)
        print(f'LIME Explanation for False Negative sample {i} using {model_name}:')
        exp.show_in_notebook(show_table=True)  # Shows the LIME explanation for the false negative instance
    return fp_indices, fn_indices


# Error Analysis and LIME for each model
for model_name, model in models.items():
    if model == models['Baseline']:
        continue  # Skip the baseline model
    print(f'Error Analysis and LIME Explanations for {model_name}:')
    
    # Perform LIME explanation on misclassified samples
    lime_explanation_for_errors(model, X_train, X_test, y_test, model_name)
    # Perform LIME explanation on false positives and false negatives
    lime_explanation_for_fp_fn(model, X_train, X_test, y_test, model_name)
    
    print('\n')

# Save the LIME explanations to file for later reference
def save_lime_explanations(model, X_train, X_test, y_test, model_name, filepath):
    # Ensure the directory exists
    if not os.path.exists(filepath):
        os.makedirs(filepath)  
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        mode='classification',
        class_names=['Negative', 'Positive'],
        feature_names=X_train.columns,
        discretize_continuous=True)
    y_pred = model.predict(X_test)
    misclassified_indices = np.where(y_pred != y_test)[0]
    explanations = {}
    for i in misclassified_indices:
        exp = explainer.explain_instance(X_test.iloc[i], model.predict_proba)
        explanations[i] = exp
        
        # Construct the file path and save the explanation
        file_path = os.path.join(filepath, f'{model_name}_explanation_{i}.html')
        try:
            exp.save_to_file(file_path)
            print(f'Saved explanation for sample {i} to {file_path}')
        except Exception as e:
            print(f"Error saving explanation for sample {i}: {e}")
    return explanations

# Example: Save explanations for the 'Random Forest' model
save_lime_explanations(models['Random Forest'], X_train, X_test, y_test, 'Random Forest', 'lime_explanations')