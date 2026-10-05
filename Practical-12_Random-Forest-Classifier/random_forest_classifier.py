"""
================================================================================
PRACTICAL NO: 12
TOPIC: Random Forest Classifier Implementation
================================================================================
Aim:
    To implement the Random Forest ensemble classifier and compare its performance with a single Decision Tree.

Objective:
    Train Random Forest with multiple trees, analyze feature importances, and compare with Decision Tree accuracy.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_wine

wine = load_wine()
X, y = wine.data, wine.target
feature_names = wine.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(X_test))
print(f"Decision Tree Accuracy:  {dt_acc:.4f}")

rf = RandomForestClassifier(n_estimators=100, max_depth=None, oob_score=True, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy:  {rf_acc:.4f}")
print(f"OOB Score:               {rf.oob_score_:.4f}")
print("\nClassification Report:\n", classification_report(y_test, rf_pred, target_names=wine.target_names))

estimators = [10, 20, 50, 100, 200]
accs = []
for n in estimators:
    rf_n = RandomForestClassifier(n_estimators=n, random_state=42)
    rf_n.fit(X_train, y_train)
    accs.append(accuracy_score(y_test, rf_n.predict(X_test)))

plt.figure(figsize=(7, 4))
plt.plot(estimators, accs, marker='o', color='darkgreen')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Random Forest: Trees vs Accuracy')
plt.grid(True, alpha=0.3)
plt.savefig('rf_feature_importances.png', dpi=300)
plt.savefig('rf_vs_dt_comparison.png', dpi=300)
plt.show()

importances = pd.Series(rf.feature_importances_, index=feature_names)
importances.nlargest(8).sort_values().plot(kind='barh', color='steelblue')
plt.title('Top 8 Feature Importances (Random Forest)')
plt.tight_layout()
plt.savefig('rf_feature_importances.png', dpi=300)
plt.savefig('rf_vs_dt_comparison.png', dpi=300)
plt.show()
