import lightgbm as lgb

# Define the objective function for Optuna for LightGBM
def objective_lgb(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'n_jobs': -1,
        'seed': 42,
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    clf = lgb.LGBMClassifier(**params)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Create a study object and optimize the objective function for LightGBM
study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=50)

# Get the best hyperparameters for LightGBM
best_params_lgb = study_lgb.best_params
print(f"Best Hyperparameters for LightGBM: {best_params_lgb}")

# Train the LightGBM classifier with the best hyperparameters
best_clf_lgb = lgb.LGBMClassifier(**best_params_lgb)
best_clf_lgb.fit(X_train, y_train)

# Make predictions on the test set
y_pred_lgb = best_clf_lgb.predict(X_test)

# Calculate and print the final evaluation metrics for LightGBM
final_accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
print(f"Final Accuracy for LightGBM with Optimal Hyperparameters: {final_accuracy_lgb:.4f}")

# Plot the confusion matrix for LightGBM
plot_confusion_matrix(best_clf_lgb, X_test, y_test, cmap=plt.cm.Blues, display_labels=['Healthy', 'Unhealthy'])
plt.title('Confusion Matrix for LightGBM (Optimal Hyperparameters)')
plt.show()
