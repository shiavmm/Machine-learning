"""
================================================================================
PRACTICAL NO: 08
TOPIC: k-Nearest Neighbors (k-NN) Classifier Implementation
================================================================================
Aim:
    To implement k-NN algorithm for classification and find the optimal value of k.

Objective:
    Apply k-NN on the Iris dataset, plot accuracy for different k values, and generate the classification report.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, :2]
y = iris.target
print("Iris Dataset - Classes:", iris.target_names)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

k_values = range(1, 20)
accuracies = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracies.append(accuracy_score(y_test, knn.predict(X_test)))

best_k = list(k_values)[np.argmax(accuracies)]
print(f"Best k: {best_k} with Accuracy: {max(accuracies):.4f}")

knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_pred = knn_best.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

plt.figure(figsize=(8, 4))
plt.plot(list(k_values), accuracies, marker='o', color='steelblue')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('k-NN: Accuracy for Different k Values')
plt.xticks(list(k_values))
plt.grid(True, alpha=0.3)
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best k={best_k}')
plt.legend()
plt.savefig('knn_accuracy_vs_k.png', dpi=300)
plt.show()
