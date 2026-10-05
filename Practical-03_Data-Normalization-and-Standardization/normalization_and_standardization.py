"""
================================================================================
PRACTICAL NO: 03
TOPIC: Data Normalization and Standardization Techniques
================================================================================
Aim:
    To apply and compare data normalization (Min-Max Scaling) and standardization (Z-score Standardization) on a dataset.

Objective:
    Understand the effect of scaling techniques on data distribution using MinMaxScaler and StandardScaler.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

data = {
    'Age':    [22, 25, 47, 52, 46, 56, 35, 29],
    'Salary': [25000, 35000, 75000, 90000, 65000, 100000, 55000, 40000],
    'Score':  [60, 72, 88, 95, 80, 98, 77, 68]
}
df = pd.DataFrame(data)
print("Original Data:\n", df)

minmax = MinMaxScaler()
df_normalized = pd.DataFrame(minmax.fit_transform(df), columns=df.columns)
print("\nNormalized Data (Min-Max):\n", df_normalized.round(4))

standard = StandardScaler()
df_standardized = pd.DataFrame(standard.fit_transform(df), columns=df.columns)
print("\nStandardized Data (Z-score):\n", df_standardized.round(4))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, title, data_plot in zip(axes,
        ['Original', 'Normalized', 'Standardized'],
        [df, df_normalized, df_standardized]):
    try:
        ax.boxplot(data_plot.values, tick_labels=data_plot.columns)
    except TypeError:
        ax.boxplot(data_plot.values, labels=data_plot.columns)
    ax.set_title(title + ' Data')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('boxplot_comparison.png', dpi=300)
plt.show()
