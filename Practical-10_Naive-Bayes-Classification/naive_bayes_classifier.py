"""
================================================================================
PRACTICAL NO: 10
TOPIC: Naive Bayes Classification
================================================================================
Aim:
    To implement Naive Bayes classification using Gaussian Naive Bayes and evaluate model performance.

Objective:
    Apply GaussianNB on the Wine dataset and interpret prediction probabilities.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target
print("Wine Dataset shape:", X.shape)
print("Classes:", wine.target_names)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=wine.target_names))

proba = gnb.predict_proba(X_test[:5])
prob_df = pd.DataFrame(proba, columns=wine.target_names)
print("\nPrediction Probabilities (first 5):\n", prob_df.round(4))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=wine.target_names)
disp.plot(cmap='Purples')
plt.title('Gaussian Naive Bayes - Confusion Matrix (Wine)')
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()
