"""
================================================================================
PRACTICAL NO: 09
TOPIC: Decision Tree Classifier Implementation
================================================================================
Aim:
    To implement a Decision Tree classifier and visualize the tree structure and feature importances.

Objective:
    Train a Decision Tree on the Iris dataset, tune max_depth, and analyze feature importances.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names   = iris.target_names.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=class_names))
print("\nTree Structure:\n", export_text(dt, feature_names=list(feature_names)))

plt.figure(figsize=(14, 6))
plot_tree(dt, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=8)
plt.title('Decision Tree - Iris Dataset')
plt.savefig('decision_tree_structure.png', dpi=300)
plt.savefig('feature_importances.png', dpi=300)
plt.show()

importances = pd.Series(dt.feature_importances_, index=feature_names)
importances.sort_values().plot(kind='barh', color='steelblue')
plt.title('Feature Importances')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('decision_tree_structure.png', dpi=300)
plt.savefig('feature_importances.png', dpi=300)
plt.show()
