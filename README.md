# classification-challenge
                                                                       Module 13 Challenge

 This project focuses on building and comparing two machine learning models, Logistic Regression and Random Forest Classifier, to classify emails as spam or not spam. The dataset used in this project is sourced from the UCI Machine Learning Repository and includes various features derived from the email text. This README outlines the steps taken in this project, from data retrieval to model evaluation.

Table of Contents

Project Overview
Retrieve the Data
Predict Model Performance
Split the Data into Training and Testing Sets
Scale the Features
Create and Fit a Logistic Regression Model
Create and Fit a Random Forest Classifier Model
Evaluate the Models
Conclusion
Project Overview
The objective of this project is to develop a machine learning pipeline to accurately classify emails as spam or not spam using Logistic Regression and Random Forest Classifier models. The project involves retrieving data, preprocessing, model training, and evaluation to determine which model performs better on the spam classification task.

Retrieve the Data
The spam dataset is available at the following URL: Spam Data CSV. This dataset includes 58 columns of features extracted from email text, with the last column indicating whether an email is spam (1) or not spam (0).

pythonCopy code
import pandas as pd # Import the data data = pd.read_csv("https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv") data.head() 

Predict Model Performance
Before creating and fitting the models, an educated guess was made regarding which model would perform better, based on the characteristics of the dataset and the nature of the models.

Split the Data into Training and Testing Sets
python
from sklearn.model_selection import train_test_split # Create the labels set `y` and features DataFrame `X` X = data.drop('spam', axis=1) y = data['spam'] # Split the data X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1) 

Scale the Features
Feature scaling was performed using StandardScaler to ensure that the model training is not biased by the scale of the features.

Python code
from sklearn.preprocessing import StandardScaler scaler = StandardScaler().fit(X_train) X_train_scaled = scaler.transform(X_train) X_test_scaled = scaler.transform(X_test) 

Create and Fit a Logistic Regression Model
python code
from sklearn.linear_model import LogisticRegression lr_model = LogisticRegression(random_state=1).fit(X_train_scaled, y_train) lr_predictions = lr_model.predict(X_test_scaled) lr_accuracy = accuracy_score(y_test, lr_predictions) print(f"Logistic Regression Model Accuracy: {lr_accuracy}")
 
Create and Fit a Random Forest Classifier Model
pythonCopy code
from sklearn.ensemble import RandomForestClassifier rf_model = RandomForestClassifier(random_state=1).fit(X_train_scaled, y_train) rf_predictions = rf_model.predict(X_test_scaled) rf_accuracy = accuracy_score(y_test, rf_predictions) print(f"Random Forest Classifier Model Accuracy: {rf_accuracy}") 

Evaluate the Models
The evaluation of the models revealed that the Random Forest Classifier model outperformed the Logistic Regression model, with an accuracy of 96.7% compared to 92.9%. This outcome was in line with the initial prediction, considering the Random Forest's ability to handle non-linear relationships and its robustness against overfitting.

Conclusion
The project successfully demonstrated the application of Logistic Regression and Random Forest Classifier models for spam classification. The Random Forest Classifier showed superior performance, highlighting its effectiveness in dealing with complex datasets. This README provides a comprehensive guide through the project's workflow, enabling replication and further exploration.


![image](https://github.com/pjpeters13/classification-challenge/assets/71742689/341ac0fb-043f-4b13-b9dd-e29b2af36ecc)

