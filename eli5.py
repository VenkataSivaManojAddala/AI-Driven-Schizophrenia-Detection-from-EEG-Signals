import eli5
import pandas as pd
import xgboost as xgb
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import train_test_split

# Load the CSV files into DataFrames
data1 = pd.read_csv('Healthy_d2_SE_8s.csv')
data2 = pd.read_csv('UnHealthy_d2_SE_8s.csv')

# Assuming the target column is named 'target'
X1 = data1.drop(columns=['target'])
y1 = data1['target']

X2 = data2.drop(columns=['target'])
y2 = data2['target']

# Concatenate the data
X = pd.concat([X1, X2], axis=0)
y = pd.concat([y1, y2], axis=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Initialize XGBoost classifier
xgb_clf = xgb.XGBClassifier(random_state=42)

# Train the classifier
xgb_clf.fit(X_train, y_train)

# Compute feature importance using ELI5
feature_names = X.columns.tolist()
feature_importance = eli5.show_weights(xgb_clf, feature_names=feature_names)
display(feature_importance)