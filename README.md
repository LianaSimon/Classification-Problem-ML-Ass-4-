# CLASSIFICATION PROBLEM

# Breast Cancer Classification - Supervised Learning Assessment

## 🎯 Objective
This project demonstrates the application of supervised learning techniques to classify tumors using the Breast Cancer Wisconsin dataset from `sklearn.datasets`.

## 📁 Dataset
- Source: `sklearn.datasets.load_breast_cancer()`
- Features: 30 numeric features related to tumor characteristics.
- Target: Binary classification - Malignant (0) or Benign (1)

## 📌 Key Steps

### 🔹 STEP 1: Loading and Preprocessing

✅ 1.1 Load the Dataset

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
Explanation:
This loads the Breast Cancer Wisconsin Diagnostic dataset, a built-in dataset from sklearn. It contains measurements of breast cell nuclei features and whether the tumor is malignant (cancerous) or benign (non-cancerous).

✅ 1.2 Create Features and Target

import pandas as pd

X = pd.DataFrame(data.data, columns=data.feature_names)  # features
y = pd.Series(data.target)  # target: 0 (malignant), 1 (benign)
Explanation:

X is the dataset with 30 numerical features (e.g., radius, texture, perimeter).

y is the label (target) for classification.

✅ 1.3 Check for Missing Values

print(X.isnull().sum().sum())
Explanation:
Before training any machine learning model, we check if there are missing (null) values. Models can't handle missing data well unless you impute (fill in) or remove them. This dataset has no missing values, so we're good.

Output

![image](https://github.com/user-attachments/assets/5b854203-aec1-4aa5-9f57-392125bfbc3c)


✅ 1.4 Feature Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Explanation:
Feature scaling is important, especially for models like SVM, k-NN, and Logistic Regression, because they are sensitive to the scale of data. StandardScaler standardizes the data so each feature has a mean = 0 and std = 1.

✅ 1.5 Train-Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
Explanation:
We split the dataset:

80% for training the models

20% for testing their accuracy
random_state=42 ensures reproducibility.


🔹 STEP 2: Classification Algorithm Implementation (5 marks)
For each model below, we will:

Train the model

Predict on the test data

Measure its accuracy

Briefly describe how it works

✅ 2.1 Logistic Regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
How it works:
It finds a line (or surface in higher dimensions) that separates classes. It outputs probabilities and chooses the class with the highest probability.

✅ 2.2 Decision Tree

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
How it works:
Splits the dataset based on feature values in a tree-like structure. Easy to interpret but can overfit on training data.

✅ 2.3 Random Forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
How it works:
Builds multiple decision trees and averages their results. It’s more accurate and less prone to overfitting than a single decision tree.

✅ 2.4 Support Vector Machine (SVM)

from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
How it works:
Finds the best hyperplane that separates the classes with the maximum margin. Works well for high-dimensional data.

✅ 2.5 k-Nearest Neighbors (k-NN)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
How it works:
Classifies based on the majority label of the k nearest data points. Sensitive to feature scaling and choice of k.

🔹 STEP 3: Model Comparison (2 marks)
✅ Compare Accuracy Scores

from sklearn.metrics import accuracy_score

results = {
    "Logistic Regression": accuracy_score(y_test, y_pred_lr),
    "Decision Tree": accuracy_score(y_test, y_pred_dt),
    "Random Forest": accuracy_score(y_test, y_pred_rf),
    "SVM": accuracy_score(y_test, y_pred_svm),
    "k-NN": accuracy_score(y_test, y_pred_knn)
}

for model, acc in results.items():
    print(f"{model}: {acc:.4f}")

    ![image](https://github.com/user-attachments/assets/60d12a81-c64b-4174-a4ad-db1f5a46b8f1)

    ![image](https://github.com/user-attachments/assets/0f2f6bd7-6404-4178-b84c-6f64b89d9e0b)


✅ Visualize Comparison

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.bar(results.keys(), results.values(), color='cornflowerblue')
plt.title("Accuracy Comparison of Classification Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.ylim(0.9, 1.0)
plt.grid(axis='y', linestyle='--')
plt.show()
Explanation:
Compare which model performed the best. In most cases, Random Forest or SVM perform best on this dataset, while k-NN may perform slightly worse depending on the data scaling and parameter tuning.

![image](https://github.com/user-attachments/assets/8a5cfdec-e49f-4879-8956-6cd3336b6615)


## 📌 Author
**Liana Simon**  
📧 lianasimon77@gmail.com  
 
