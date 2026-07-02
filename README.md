# 🏦 Loan Approval Prediction using Machine Learning

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![Flask](https://img.shields.io/badge/Flask-Web_App-black?style=for-the-badge&logo=flask)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

---

# Abstract

Loan approval is one of the most critical decision-making processes in the banking and financial sector. Financial institutions must evaluate numerous applicant attributes before approving or rejecting a loan application. Manual assessment is often time-consuming and susceptible to human bias.

This project presents a Machine Learning-based Loan Approval Prediction System that automates the decision-making process by analyzing applicant information such as annual income, loan amount, CIBIL score, education level, employment status, number of dependents, and asset values.

Multiple supervised machine learning algorithms are trained and evaluated to identify the best-performing classifier. A Flask web application provides an intuitive interface for real-time predictions.

---

# Objectives

- Build an end-to-end Machine Learning pipeline.
- Compare multiple classification algorithms.
- Perform data preprocessing and feature engineering.
- Evaluate model performance using multiple metrics.
- Deploy the trained model using Flask.
- Create an interactive web interface for prediction.

---

# Problem Statement

Banks receive thousands of loan applications every year.

Approving a loan for an unsuitable applicant increases financial risk, while rejecting a qualified applicant results in lost business opportunities.

The objective of this project is to predict whether a loan application should be **Approved** or **Rejected** based on historical applicant data.

---

# Dataset

The dataset contains applicant financial information including:

| Feature | Description |
|----------|-------------|
| Number of Dependents | Family dependents |
| Education | Graduate / Not Graduate |
| Self Employed | Yes / No |
| Annual Income | Applicant annual income |
| Loan Amount | Requested loan amount |
| Loan Term | Loan duration |
| CIBIL Score | Credit score |
| Residential Assets | Residential property value |
| Commercial Assets | Commercial property value |
| Luxury Assets | Luxury asset value |
| Bank Assets | Total bank assets |
| Loan Status | Target Variable |

---

# Machine Learning Workflow

```
Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Missing Value Handling
      │
      ▼
Label Encoding
      │
      ▼
Feature Scaling
      │
      ▼
Train-Test Split
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Best Model Selection
      │
      ▼
Model Deployment
```

---

# Algorithms Used

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- Voting Classifier

---

# Data Preprocessing

The following preprocessing techniques were implemented:

- Removing unnecessary columns
- Missing value imputation
- Label Encoding
- Standard Feature Scaling
- Train-Test Split

---

# Model Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

---

# Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Pandas | Data Analysis |
| NumPy | Numerical Computing |
| Scikit-Learn | Machine Learning |
| Joblib | Model Serialization |
| Flask | Web Framework |
| HTML5 | Frontend |
| CSS3 | Styling |

---

# Project Structure

```
Loan_Approval_Prediction
│
├── data
│     └── loan_approval_dataset.csv
│
├── templates
│     └── index.html
│
├── static
│     └── style.css
│
├── train_model.py
├── predict.py
├── app.py
├── model.pkl
├── scaler.pkl
├── label_encoders.pkl
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Loan_Approval_Prediction.git
```

Move into the project directory

```bash
cd Loan_Approval_Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Train the model

```bash
python train_model.py
```

Run the Flask application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000
```

---

# Features

- Multiple Machine Learning Algorithms
- Automatic Data Preprocessing
- Model Comparison
- Interactive User Interface
- Real-Time Loan Prediction
- Responsive Design
- Clean Project Structure
- Easy Deployment

---

# Future Improvements

- Hyperparameter Optimization using GridSearchCV
- Explainable AI using SHAP
- Probability Score Visualization
- Model Confidence Estimation
- Database Integration
- Cloud Deployment
- User Authentication
- Loan Risk Dashboard

---

# Results

The trained models were evaluated on unseen test data, and the best-performing classifier was selected based on evaluation metrics. The application provides real-time predictions for new loan applications through an intuitive web interface.

---

# Conclusion

This project demonstrates the application of supervised machine learning techniques to automate loan approval prediction. By combining data preprocessing, multiple classification algorithms, comparative evaluation, and web deployment, the system offers an effective and scalable solution for assisting financial institutions in decision-making.

The modular architecture also enables future enhancements such as advanced ensemble methods, explainable AI, and cloud deployment.

---

# Author

**Sukalpa Manna**

Machine Learning Enthusiast | Data Science | Artificial Intelligence

GitHub: https://github.com/yourusername

---

# License

This project is licensed under the MIT License.

---

⭐ If you found this project useful, consider giving it a star.
