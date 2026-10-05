"""
================================================================================
PRACTICAL NO: 04
TOPIC: Implementation of Simple Linear Regression
================================================================================
Aim:
    To implement Simple Linear Regression to predict a continuous output variable based on a single input feature.

Objective:
    Build a linear regression model using Scikit-learn, evaluate using R² score and MSE, and visualize the regression line.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Exam_Score':    [35, 45, 50, 60, 65, 70, 75, 85, 90, 95]
}
df = pd.DataFrame(data)
print("Dataset:\n", df)

X = df[['Hours_Studied']]
y = df['Exam_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print(f"\nIntercept: {model.intercept_:.4f}")
print(f"Slope:     {model.coef_[0]:.4f}")

y_pred = model.predict(X_test)
print("\nPredictions vs Actual:")
for pred, actual in zip(y_pred, y_test):
    print(f"  Predicted: {pred:.2f}  |  Actual: {actual}")

mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"\nMSE: {mse:.4f}")
print(f"R2:  {r2:.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Actual Data', s=60)
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
plt.title('Simple Linear Regression: Hours vs Score')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('regression_line.png', dpi=300)
plt.show()
