from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
import pandas as pd
import optuna

# Load the CSV files into DataFrames (replace file names as needed)
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

# Define evaluation metric and StratifiedKFold
scorer = 'accuracy'  # You can choose the evaluation metric (e.g., 'accuracy', 'precision', 'recall', 'f1')
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def objective(trial):
    # Define hyperparameters to tune
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_uniform('gamma', 0.0, 1.0),
    }

    # Initialize XGBClassifier with tuned hyperparameters
    clf = XGBClassifier(**params)

    # Perform cross-validation
    accuracy_scores = cross_val_score(clf, X, y, cv=cv, scoring=scorer)

    # Return the negative mean accuracy (Optuna maximizes, so we want to minimize -accuracy)
    return -accuracy_scores.mean()

# Perform hyperparameter optimization with Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Print the best hyperparameters found by Optuna
print("Best Hyperparameters:", study.best_params)

# Initialize XGBClassifier with the best hyperparameters
best_params = study.best_params
best_clf = XGBClassifier(**best_params)

# Perform cross-validation with the best classifier
best_accuracy_scores = cross_val_score(best_clf, X, y, cv=cv, scoring=scorer)

# Print cross-validation results
print(f"\nCross-Validation {scorer} Scores with Optimized Hyperparameters:", best_accuracy_scores)
print(f"Mean {scorer} Score with Optimized Hyperparameters:", best_accuracy_scores.mean())
