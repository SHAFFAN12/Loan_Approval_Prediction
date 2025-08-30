# **Loan Approval Prediction**

This project predicts whether a loan application will be approved. It focuses on handling missing values, encoding categorical features, dealing with imbalanced classes (SMOTE), and evaluating model performance using precision, recall, and F1-score.

## 📌 Project Overview
- **Objective:** Predict loan approval (binary classification) and optimize for metrics relevant to imbalanced data (precision, recall, F1-score).  
- **Dataset:** Loan Approval Prediction (Kaggle)  
- **Approach:** Data cleaning → encoding → outlier handling → scaling → SMOTE → model training & evaluation.

## 🛠️ Tools & Libraries
- Python  
- pandas, numpy  
- scikit-learn  
- imbalanced-learn (SMOTE)  
- matplotlib, seaborn


## ⚙️ Pipeline / Steps
1. Load dataset and inspect distributions & missing values.  
2. Fill missing categorical values with mode; numeric with median.  
3. Convert `Dependents` and encode binary categorical variables (LabelEncoder).  
4. One-hot encode `Property_Area`. Drop `Loan_ID`.  
5. Handle outliers using IQR capping.  
6. Split into train/test and scale features (StandardScaler).  
7. Apply **SMOTE** on training data to address class imbalance.  
8. Train models: **Logistic Regression** and **Decision Tree**.  
9. Evaluate using: Accuracy, Precision, Recall, F1-score, Confusion Matrix.  
10. Compare models and choose best by F1-score (or recall depending on risk).


## 📈 Example Results
(Replace with your actual outputs)
- Logistic Regression → Precision: 0.707, Recall:0.512, F1-score:0.594  
- Decision Tree → Precision: 0.646, Recall: 0.638, F1-score: 0.642

**Best model:** Replace with model name and final F1-score.



## 🚀 How to run
1. Clone the repo:
```bash
git clone https://github.com/SHAFFAN12/loan-approval-prediction.git
cd loan-approval-prediction
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open loan_approval_prediction.ipynb and run cells.

4. To reproduce model training, run the notebook from top → bottom.

