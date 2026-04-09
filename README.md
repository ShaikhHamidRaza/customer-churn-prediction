# Customer Churn Prediction using Machine Learning

Customer churn prediction is an important problem for telecom companies.  
Losing customers directly affects business revenue, so companies try to identify customers who are likely to leave their service.

In this project, I built a machine learning model to predict whether a telecom customer will churn or not using historical customer data.

---

# Project Overview

The goal of this project is to analyze telecom customer data and develop a machine learning model that can predict customer churn.

By identifying customers who are likely to churn, companies can take proactive steps such as offering discounts, improving services, or providing personalized support to retain them.

---

# Dataset

The dataset used in this project is the **Telco Customer Churn dataset**.

Dataset information:

- Total records: ~7000 customers
- Target variable: **Churn (Yes / No)**

Some important features include:

- Tenure (how long the customer has stayed with the company)
- Monthly Charges
- Total Charges
- Contract Type
- Internet Service
- Payment Method

---

# Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand patterns and relationships in the data.

Key analyses included:

- Churn distribution
- Contract type vs churn
- Monthly charges vs churn
- Tenure vs churn
- Correlation heatmap

These visualizations help identify important factors affecting customer churn.

---

# Data Preprocessing

Before training the models, the following preprocessing steps were applied:

- Removed unnecessary columns such as `customerID`
- Converted `TotalCharges` to numeric format
- Handled missing values
- Encoded categorical variables
- Applied feature scaling where required

---

# Handling Class Imbalance

Customer churn datasets are often imbalanced.

To address this problem, **SMOTE (Synthetic Minority Over-sampling Technique)** was used to balance the dataset.

This helps the model learn patterns from both churn and non-churn customers.

---

# Machine Learning Models Used

Multiple machine learning models were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

Each model was evaluated using standard classification metrics.

---

# Model Evaluation

Models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

These metrics help measure how well the model predicts customer churn.

---

# Best Model

After comparing all models, **Logistic Regression performed the best**.

Performance:

Accuracy: ~82%  
Precision: ~0.68  
Recall: ~0.60  
F1 Score: ~0.64  
ROC-AUC: ~0.75

The model shows a balanced ability to identify churn customers.

---

# Feature Importance

Some of the key factors influencing customer churn include:

- Contract type
- Monthly charges
- Tenure
- Internet service

Understanding these factors can help businesses design better customer retention strategies.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- XGBoost
- Imbalanced-learn (SMOTE)

---

# Project Structure

customer-churn-prediction
│
├── Customer_Churn_Prediction.ipynb
├── churn_model.pkl
├── requirements.txt
└── README.md


---

# Future Improvements

Possible improvements for this project include:

- Hyperparameter tuning for better performance
- Model explainability using SHAP
- Building a Streamlit web app for churn prediction
- Deploying the model as an API

---

# Author

Hamid Raza  
BCA Student | Aspiring AI/ML Engineer  

GitHub  
https://github.com/ShaikhHamidRaza

LinkedIn  
https://linkedin.com/in/shaikhhamidraza
