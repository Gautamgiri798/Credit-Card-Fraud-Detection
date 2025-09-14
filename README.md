# Credit-Card-Fraud-Detection
This project focuses on detecting fraudulent credit card transactions using machine learning. The primary goal is to build and evaluate several classification models to accurately identify fraudulent transactions from a highly imbalanced dataset.


📖 Project Overview
The project follows a standard data science pipeline:

1.Exploratory Data Analysis (EDA): The dataset is loaded, inspected, and visualized to understand its structure, feature distributions, and the severe class imbalance between fraudulent and legitimate transactions.

2.Data Preprocessing: Features such as Time and Amount are scaled to prevent them from dominating the model training process. The class imbalance is addressed to ensure the models can learn from the minority class (fraudulent transactions).

3.Model Training: Several machine learning classifiers are trained on the preprocessed data.

4.Model Evaluation & Comparison: The performance of each model is rigorously evaluated using metrics appropriate for imbalanced datasets, such as Precision, Recall, F1-Score, and ROC-AUC. Finally, the models are compared to identify the most effective one for this task.


📊 Dataset
The dataset contains credit card transactions made in September 2013 by European cardholders. It presents transactions that occurred in two days, with 492 frauds out of 284,807 transactions.

Key characteristics:

  The dataset is highly imbalanced, with the positive class (frauds) accounting for only 0.172% of all transactions.

  It contains only numerical input variables which are the result of a PCA transformation. The original features are not provided due to confidentiality issues.

  Features V1, V2, ... V28 are the principal components obtained with PCA.

  The only features which have not been transformed with PCA are Time and Amount.

  The Class feature is the response variable, where it takes a value of 1 in case of fraud and 0 otherwise.


⚙️ Project Pipeline
The notebook Credit Card Fraud Detection.ipynb is structured as follows:

1.Import Libraries: Essential Python libraries like Pandas, NumPy, Matplotlib, Seaborn, Plotly, and Scikit-learn are loaded.

2.Load Data: The credit card dataset is loaded into a Pandas DataFrame.

3.Exploratory Data Analysis:

  Initial data inspection (head(), describe(), info()).

  Checking for missing values.

  Visualizing the class distribution to highlight the imbalance.

  Analyzing the distributions of the Time and Amount features.

4.Data Preprocessing:

  Scaling the Amount and Time columns using StandardScaler or RobustScaler.

  Splitting the data into training and testing sets.

  Handling the class imbalance (e.g., using SMOTE or by selecting models robust to imbalance).

5.Model Implementation:

  Training various classification algorithms (such as Logistic Regression, Decision Trees, Random Forest, etc.).

6.Performance Evaluation:

  Evaluating each model using metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

  Generating confusion matrices for a detailed view of prediction results.

  Plotting ROC curves.

7.Results Comparison:

  Creating summary tables and bar plots to compare the performance metrics and training times of all the trained models.


🤖 Models Trained
The following classification models were trained and evaluated:

  Logistic Regression

  Decision Tree Classifier

  Random Forest Classifier

  Other classifiers as implemented in the notebook...


📈 Results Summary
The performance of each model is systematically measured and compared. The results, including detailed metrics and training times, are presented in a summary DataFrame and visualized through plots. This comparative analysis helps in determining the most suitable model that balances predictive accuracy with computational efficiency for detecting credit card fraud.


🚀 How to Run this Project
To run this project on your local machine, follow these steps:

1.Clone the repository:

  git clone https://github.com/Gautamgiri798/Credit-Card-Fraud-Detection

2.Navigate to the project directory:

  cd <project-directory>

3.Install the required dependencies:
  It is recommended to use a virtual environment.

  pip install -r requirements.txt

  If a requirements.txt is not available, you can install the packages manually:

  pip install pandas numpy matplotlib seaborn plotly scikit-learn jupyter

4.Launch Jupyter Notebook:

  jupyter notebook

5.Run the notebook:
  Open the Credit Card Fraud Detection.ipynb file and execute the cells sequentially.
