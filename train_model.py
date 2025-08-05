import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier, plot_importance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Load dataset
df = pd.read_csv("cleaned_encoded_data.csv")

# Separate features and target
X = df.drop("readmitted_binary", axis=1)
y = df["readmitted_binary"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------------------
# Apply SMOTE on training set only
# ----------------------------

print("Before SMOTE:", np.bincount(y_train))

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("After SMOTE:", np.bincount(y_train_res))

# ----------------------------
# Define base XGBoost model for feature selection and classification
# ----------------------------
xgb_selector = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

selector = SelectFromModel(estimator=xgb_selector, threshold='median')

xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    verbosity=0
)

# ----------------------------
# Hyperparameter Grid
# ----------------------------
param_grid = {
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [3, 5, 7],
    'xgb__learning_rate': [0.05, 0.1, 0.2],
    'xgb__subsample': [0.8, 1.0],
    'xgb__colsample_bytree': [0.8, 1.0],
}

# ----------------------------
# Build pipeline with SMOTE + Feature Selection + XGBoost
# ----------------------------
pipeline = Pipeline([
    ('selector', selector),      # Feature selection step
    ('xgb', xgb)                 # Final classifier
])

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1_weighted',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# ----------------------------
# Train model
# ----------------------------
grid_search.fit(X_train_res, y_train_res)

best_model = grid_search.best_estimator_

# ----------------------------
# Predict on test set
# ----------------------------
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# ----------------------------
# Evaluation
# ----------------------------
print("\nTuned XGBoost Model with SMOTE + Feature Selection:")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------
# Feature Importance (only top features used in final model)
# ----------------------------
plt.figure(figsize=(12, 6))
plot_importance(best_model.named_steps['xgb'], max_num_features=15, importance_type='gain')
plt.title("Top 15 Feature Importances (after selection)")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# ----------------------------
# Save best model
# ----------------------------
joblib.dump(best_model, "xgboost_readmission_model.pkl")
print("\nModel saved as xgboost_readmission_model.pkl")
