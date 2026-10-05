# Practical 02: Data Preprocessing: Handling Missing Values and Encoding Categorical Variables

## 🎯 Aim
To understand and implement data preprocessing techniques — handling missing values using imputation and encoding categorical variables using Label Encoding and One-Hot Encoding.

## 📌 Objective
Apply preprocessing steps on a sample dataset to prepare it for machine learning by treating null values and converting non-numeric features to numeric form.

## 📖 Overview & Theory
Demonstrates essential data preprocessing techniques on tabular data, including handling missing numerical and categorical values using Scikit-Learn's `SimpleImputer` (mean and most-frequent strategies), as well as converting categorical variables to numerical features using `LabelEncoder` and pandas `get_dummies` (One-Hot Encoding).

---

## 💻 Source Code
The practical implementation is available in [`data_preprocessing.py`](./data_preprocessing.py).

```python
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
```

---

## 📊 Output & Results

### Console Output
```text
Original DataFrame:
     Age   Salary  Gender    City
0  25.0  50000.0    Male  Mumbai
1   NaN  60000.0  Female    Pune
2  30.0      NaN  Female   Delhi
3  22.0  45000.0     NaN  Mumbai
4   NaN  70000.0    Male     NaN
5  28.0      NaN  Female    Pune

Missing values:
 Age       2
Salary    2
Gender    1
City      1
dtype: int64

After Imputation:
      Age   Salary  Gender    City
0  25.00  50000.0    Male  Mumbai
1  26.25  60000.0  Female    Pune
2  30.00  56250.0  Female   Delhi
3  22.00  45000.0  Female  Mumbai
4  26.25  70000.0    Male  Mumbai
5  28.00  56250.0  Female    Pune

Label Encoded Gender: [1 0 0 0 1 0]

After One-Hot Encoding:
      Age   Salary  Gender  Gender_Encoded  City_Delhi  City_Mumbai  City_Pune
0  25.00  50000.0    Male               1       False         True      False
1  26.25  60000.0  Female               0       False        False       True
2  30.00  56250.0  Female               0        True        False      False
3  22.00  45000.0  Female               0       False         True      False
4  26.25  70000.0    Male               1       False         True      False
5  28.00  56250.0  Female               0       False        False       True
```



---

## 🏁 Conclusion / Result
Missing values were filled using mean and mode imputation. Label Encoding and One-Hot Encoding were applied successfully to convert categorical data into numeric format.

---

## 🚀 How to Run
Execute the script using Python:
```bash
python data_preprocessing.py
```
