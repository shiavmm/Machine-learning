"""
================================================================================
PRACTICAL NO: 02
TOPIC: Data Preprocessing: Handling Missing Values and Encoding Categorical Variables
================================================================================
Aim:
    To understand and implement data preprocessing techniques — handling missing values using imputation and encoding categorical variables using Label Encoding and One-Hot Encoding.

Objective:
    Apply preprocessing steps on a sample dataset to prepare it for machine learning by treating null values and converting non-numeric features to numeric form.
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

data = {
    'Age':    [25, np.nan, 30, 22, np.nan, 28],
    'Salary': [50000, 60000, np.nan, 45000, 70000, np.nan],
    'Gender': ['Male', 'Female', 'Female', np.nan, 'Male', 'Female'],
    'City':   ['Mumbai', 'Pune', 'Delhi', 'Mumbai', np.nan, 'Pune']
}
df = pd.DataFrame(data)
print("Original DataFrame:\n", df)
print("\nMissing values:\n", df.isnull().sum())

num_imputer = SimpleImputer(strategy='mean')
df[['Age', 'Salary']] = num_imputer.fit_transform(df[['Age', 'Salary']])
cat_imputer = SimpleImputer(strategy='most_frequent')
df[['Gender', 'City']] = cat_imputer.fit_transform(df[['Gender', 'City']])
print("\nAfter Imputation:\n", df)

le = LabelEncoder()
df['Gender_Encoded'] = le.fit_transform(df['Gender'])
print("\nLabel Encoded Gender:", df['Gender_Encoded'].values)

df_encoded = pd.get_dummies(df, columns=['City'], drop_first=False)
print("\nAfter One-Hot Encoding:\n", df_encoded)
