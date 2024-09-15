# Fraud Detection Analysis

## Overview
This project involves analyzing a dataset to detect fraudulent transactions. The dataset contains various features related to transactions, such as transaction amount, merchant details, and whether the transaction was fraudulent.

## Libraries Used
- `pandas`: For data manipulation and analysis.
- `matplotlib.pyplot`: For creating plots and visualizations.
- `seaborn`: For making statistical graphics and enhancing plots.
- `scikit-learn`: For model training, data preprocessing, and evaluation.

## Data Loading and Initial Exploration
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from a CSV file
data = pd.read_csv('Dataset/fraudTrain.csv')

# Display the first few rows of the dataset to get an initial look at the data
print(data.head())

# Get summary statistics of the dataset to understand the distribution of numerical features
print(data.describe())

# Check for missing values in the dataset to identify any data cleaning needs
print(data.isnull().sum())
```

## Data Visualization
### Distribution of Transaction Amounts
```python
# Plot the distribution of transaction amounts to visualize how transaction amounts are spread
plt.figure(figsize=(10, 6))
sns.histplot(data['amt'], bins=50, kde=True)
plt.title('Distribution of Transaction Amounts')
plt.show()
```

### Fraud vs Legitimate Transactions
```python
# Plot the distribution of fraud vs. legitimate transactions to see the class imbalance
plt.figure(figsize=(6, 4))
sns.countplot(data['is_fraud'])
plt.title('Fraud vs Legitimate Transactions')
plt.show()
```

## Data Cleaning and Feature Engineering
```python
# Drop any rows with missing values (if applicable)
data_cleaned = data.dropna()

# Convert the transaction date to datetime for time-based analysis
data_cleaned['trans_date_trans_time'] = pd.to_datetime(data_cleaned['trans_date_trans_time'])

# Extract useful features from datetime (hour, day, month, year)
data_cleaned['hour'] = data_cleaned['trans_date_trans_time'].dt.hour
data_cleaned['day'] = data_cleaned['trans_date_trans_time'].dt.day
data_cleaned['month'] = data_cleaned['trans_date_trans_time'].dt.month
data_cleaned['year'] = data_cleaned['trans_date_trans_time'].dt.year

# Convert categorical columns like merchant and category into dummy/one-hot encoded columns
data_cleaned = pd.get_dummies(data_cleaned, columns=['merchant', 'category'], drop_first=True)

# Drop unnecessary columns (e.g., customer names)
data_cleaned = data_cleaned.drop(['first', 'last', 'street', 'city', 'state', 'job', 'dob'], axis=1)
```

## Model Training and Evaluation
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Define the feature set (X) and the target (y)
X = data_cleaned.drop(['is_fraud'], axis=1)
y = data_cleaned['is_fraud']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data (helps with convergence in some algorithms)
scaler = StandardScaler()

# Fit and transform the numeric columns only (assuming all features are numeric except categorical ones)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict the outcomes on the test set
y_pred = rf_model.predict(X_test)

# Print evaluation metrics
print(classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('AUC-ROC:', roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1]))
```

## Results
The model achieved the following performance metrics:
- **Precision**: 1.00 for legitimate transactions, 0.97 for fraudulent transactions.
- **Recall**: 1.00 for legitimate transactions, 0.55 for fraudulent transactions.
- **F1-Score**: 1.00 for legitimate transactions, 0.70 for fraudulent transactions.
- **Accuracy**: 1.00 overall.
- **AUC-ROC**: 0.98

## Conclusion
The Random Forest model performed well in detecting fraudulent transactions, with high precision and accuracy. However, the recall for fraudulent transactions indicates room for improvement, suggesting the need for further tuning or additional features to better capture fraudulent behavior. #AppreciateWealthDataScience
