from sklearn.ensemble import RandomForestClassifier

# Define the objective function for Optuna for Random Forest
def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_float('min_samples_split', 0.1, 1.0),
        'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 0.5),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0),
        'random_state': 42,
    }

    clf = RandomForestClassifier(**params)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Create a study object and optimize the objective function for Random Forest
study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=50)

# Get the best hyperparameters for Random Forest
best_params_rf = study_rf.best_params
print(f"Best Hyperparameters for Random Forest: {best_params_rf}")

# Train the Random Forest classifier with the best hyperparameters
best_clf_rf = RandomForestClassifier(**best_params_rf)
best_clf_rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = best_clf_rf.predict(X_test)

# Calculate and print the final evaluation metrics for Random Forest
final_accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Final Accuracy for Random Forest with Optimal Hyperparameters: {final_accuracy_rf:.4f}")

# Plot the confusion matrix for Random Forest
plot_confusion_matrix(best_clf_rf, X_test, y_test, cmap=plt.cm.Blues, display_labels=['Healthy', 'Unhealthy'])
plt.title('Confusion Matrix for Random Forest (Optimal Hyperparameters)')
plt.show()
