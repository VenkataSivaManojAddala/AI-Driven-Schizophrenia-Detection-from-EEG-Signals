#3s data
import lazypredict
### importing lazypredict library
import lazypredict
### importing LazyClassifier for classification problem
from lazypredict.Supervised import LazyClassifier
### importing LazyClassifier for classification problem because here we are solving Classification use case.
from lazypredict.Supervised import LazyClassifier
### spliting dataset into training and testing part
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the CSV files into DataFrames (replace file names as needed)
data1 = pd.read_csv('Healthy_d2_SE_8s.csv')
data2 = pd.read_csv('UnHealthy_d2_SE_8s.csv')

# Initialize lists to store metrics for each iteration
accuracy_list = []
precision_list = []
f1_list = []
recall_list = []
specificity_list = []

# Repeat the process 10 times with different random states
for i in range(1):
    # Assuming the target column is named 'target'
    X1 = data1.drop(columns=['target'])
    y1 = data1['target']

    X2 = data2.drop(columns=['target'])
    y2 = data2['target']

    # Split the data into training and testing sets with different random states for each iteration
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

    # Concatenate the training and testing sets
    X_train = pd.concat([X_train1, X_train2], axis=0)
    X_test = pd.concat([X_test1, X_test2], axis=0)
    y_train = pd.concat([y_train1, y_train2], axis=0)
    y_test = pd.concat([y_test1, y_test2], axis=0)

    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric = None)

### fitting data in LazyClassifier
    models,predictions = clf.fit(X_train, X_test, y_train, y_test)
    print(models)