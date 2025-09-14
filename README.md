💳 Credit Card Fraud Detection

Detect fraudulent credit card transactions using advanced machine learning techniques! This project trains and compares multiple classifiers to accurately identify fraudulent transactions in a highly imbalanced dataset.


📖 Project Overview

This project follows a standard data science pipeline to build an effective credit card fraud detection system:

Exploratory Data Analysis (EDA)
Understand the dataset structure, distributions, and severe class imbalance.

Data Preprocessing
Scale features like Time and Amount and handle class imbalance using SMOTE or robust models.

Model Training
Train multiple machine learning classifiers including Logistic Regression, Decision Trees, Random Forest, and more.

Model Evaluation & Comparison
Evaluate models with metrics suitable for imbalanced datasets:

Precision, Recall, F1-Score

ROC-AUC

Confusion Matrix
Compare models to identify the most effective one.

📊 Dataset

Transactions made by European cardholders in September 2013.

Total transactions: 284,807 | Frauds: 492 (0.172%).

Features V1–V28 are PCA components.

Original features not disclosed due to confidentiality.

Time and Amount are the only non-PCA features.

Target variable Class:

0 → Legitimate

1 → Fraud

⚙️ Project Pipeline

Import Libraries
Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-learn

Load Data
Load dataset into a Pandas DataFrame

EDA

Initial data inspection (head(), describe(), info())

Check for missing values

Visualize class imbalance

Analyze Time and Amount distributions

Data Preprocessing

Scale Amount and Time

Split data into training/testing sets

Handle class imbalance with SMOTE or robust models

Model Implementation
Train classifiers like:

Logistic Regression

Decision Tree

Random Forest

Gradient Boosting / XGBoost (optional)

Performance Evaluation

Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

Confusion Matrices & ROC curves

Results Comparison

Summary tables and bar plots

Compare metrics and training times

🤖 Models Trained

| Model                       | Precision | Recall | F1-Score | ROC-AUC |
| --------------------------- | --------- | ------ | -------- | ------- |
| Logistic Regression         | ✅         | ✅      | ✅        | ✅       |
| Decision Tree Classifier    | ✅         | ✅      | ✅        | ✅       |
| Random Forest Classifier    | ✅         | ✅      | ✅        | ✅       |
| Gradient Boosting / XGBoost | ✅         | ✅      | ✅        | ✅       |

⚡ Detailed metrics and visualizations are available in the notebook.

🚀 How to Run

Clone the repository

git clone https://github.com/Gautamgiri798/Credit-Card-Fraud-Detection.git


Navigate to the project directory

cd Credit-Card-Fraud-Detection


Install dependencies

pip install -r requirements.txt


Launch Jupyter Notebook

jupyter notebook


Run the notebook
Open Credit Card Fraud Detection.ipynb and run all cells sequentially.

