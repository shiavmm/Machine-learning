# Machine Learning Laboratory (Sem V) 🤖

Comprehensive laboratory journal and practical implementations for the **Machine Learning** course, Bachelor of Science in Data Science (Semester V, Academic Year 2026–2027), **KES' Shroff College of Arts and Commerce** (Autonomous), Kandivali (W), Mumbai.

**Student:** Shivam Pal  
**Roll No / Seat No:** TDDS056A  
**Course:** B.Sc. Data Science (TYBSc) — Semester V  
**Subject:** Machine Learning  

---

## 📑 Repository Contents & Practical Index

Each practical is organized **topic-wise** into dedicated directories containing the source code, dataset handling, comprehensive explanation, terminal output, and high-resolution visualizations.

| Sr. No. | Practical Topic | Directory | Key Algorithm / Technique | Primary Dataset / Task | Output & Plots |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **02** | [Data Preprocessing](./Practical-02_Data-Preprocessing) | `Practical-02_Data-Preprocessing/` | Mean/Mode Imputation, Label Encoding, One-Hot Encoding | Tabular Sample Data | Encoded Matrices |
| **03** | [Data Normalization & Standardization](./Practical-03_Data-Normalization-and-Standardization) | `Practical-03_Data-Normalization-and-Standardization/` | Min-Max Normalization (`MinMaxScaler`), Z-Score (`StandardScaler`) | Numerical Feature Distribution | Box Plot Distribution |
| **04** | [Simple Linear Regression](./Practical-04_Simple-Linear-Regression) | `Practical-04_Simple-Linear-Regression/` | Ordinary Least Squares Linear Regression | Study Hours vs. Exam Score | Best-Fit Regression Line |
| **05** | [Multiple Linear Regression](./Practical-05_Multiple-Linear-Regression) | `Practical-05_Multiple-Linear-Regression/` | Multi-variable Linear Regression, MSE, RMSE, R² | House Price Prediction | Actual vs. Predicted Plot |
| **06** | [Polynomial Regression](./Practical-06_Polynomial-Regression) | `Practical-06_Polynomial-Regression/` | Non-Linear Fitting via `PolynomialFeatures` & Pipelines | Synthetic Non-linear Curve | Degree Comparison Curve |
| **07** | [Logistic Regression](./Practical-07_Logistic-Regression) | `Practical-07_Logistic-Regression/` | Sigmoid-based Binary Classification, Confusion Matrix | Breast Cancer Wisconsin Dataset | Confusion Matrix Heatmap |
| **08** | [k-Nearest Neighbors (k-NN)](./Practical-08_k-Nearest-Neighbors) | `Practical-08_k-Nearest-Neighbors/` | Instance-based k-NN Classifier, Hyperparameter Tuning (k=1..19) | Fisher's Iris Dataset | Accuracy vs. k Curve |
| **09** | [Decision Tree Classifier](./Practical-09_Decision-Tree-Classifier) | `Practical-09_Decision-Tree-Classifier/` | CART Decision Tree, Gini Impurity, Tree Export | Fisher's Iris Dataset | Tree Structure & Feature Importances |
| **10** | [Naive Bayes Classification](./Practical-10_Naive-Bayes-Classification) | `Practical-10_Naive-Bayes-Classification/` | Gaussian Naive Bayes, Posterior Probabilities | Multi-class Wine Dataset | Confusion Matrix & Probabilities |
| **12** | [Random Forest Ensemble](./Practical-12_Random-Forest-Classifier) | `Practical-12_Random-Forest-Classifier/` | Bagging Ensemble, Out-Of-Bag (OOB) Scoring, Variance Reduction | Multi-class Wine Dataset | Feature Ranking & RF vs DT Accuracy |

---

## 📄 Complete Journal Document

The extracted official Machine Learning laboratory journal pages (Cover Page, Certificate, Index, and Practicals 2–12) from the practical blackbook are available as:
- 📖 **[`Machine_Learning_Practicals.pdf`](./Machine_Learning_Practicals.pdf)** *(23 Pages)*

---

## 🛠️ Environment Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/shiavmm/Machine-learning.git
cd Machine-learning
```

### 2. Create and Activate Virtual Environment (Optional)
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Running Any Practical
Navigate to the desired practical directory or run directly from the root:
```bash
# Example: Running Practical 04 (Simple Linear Regression)
python Practical-04_Simple-Linear-Regression/simple_linear_regression.py

# Example: Running Practical 09 (Decision Tree)
python Practical-09_Decision-Tree-Classifier/decision_tree_classifier.py
```

---

## 📚 Technologies & Libraries
- **Language:** Python 3.10+
- **Data Manipulation:** NumPy, Pandas
- **Machine Learning:** Scikit-Learn (`sklearn`)
- **Data Visualization:** Matplotlib

---
*Created as part of the TYBSc Data Science curriculum, University of Mumbai.*
