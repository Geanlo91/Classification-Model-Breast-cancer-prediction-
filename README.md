# Breast Cancer Prediction Project

This project is a comprehensive pipeline for predicting breast cancer using machine learning models. It encompasses data preprocessing, model training, evaluation, interpretability, and a user-friendly GUI for real-time predictions.

## Overview 

This project aims to predict whether breast cancer is benign or malignant using machine learning models. It includes:

- Data preprocessing and feature engineering.
- Training and evaluation of multiple models (Logistic Regression, Decision Tree, Random Forest, KNN, etc.).
- Model interpretability using LIME and SHAP.
- A GUI for real-time cancer predictions.

## Dataset Description 

The project uses a dataset with features representing various physical attributes of cell nuclei extracted from breast masses. The target variable is a diagnosis (0 for Benign, 1 for Malignant).

## Python Libraries 

Ensure you have the following Python libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- pycaret
- lime
- shap
- joblib
- tkinter

Install them using:
pip install pandas numpy matplotlib seaborn scikit-learn pycaret lime shap joblib

## How to Use
1. Preprocessing and Training
Run the first script to preprocess the data and save preprocessed_data.csv.
Execute the second script to train models and save them as .joblib files.

2. Model Evaluation
Use the third script to evaluate the models, visualize performance, and analyze errors.

3. Real-Time Prediction
Run the fourth script to launch the GUI for real-time predictions:
bash

python breast_cancer_gui.py

Input feature values into the GUI and click Predict to see the result.