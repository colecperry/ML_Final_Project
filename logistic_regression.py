import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Read the pre-split files: take our preprocessed CSV files and load them into DataFrames
train_df = pd.read_csv("train_split.csv")
test_df  = pd.read_csv("test_split.csv")

# 2) Separate text and metadata (publisher_id and date_ordinal) from each DataFrame
X_train_df = train_df
X_test_df  = test_df

y_train = train_df["label"]
y_test  = test_df["label"]

# 3-5) Turn text into TF-IDF vectors (default parameters), scale the metadata, combine text features and metadata into one array handled in ColumnTransformer
preprocessor_default = ColumnTransformer([
    ('text', TfidfVectorizer(), 'combined_text'),
    ('meta', StandardScaler(), ['publisher_id', 'date_ordinal'])
])

# Build a single pipeline for training & evaluation
pipeline_default = Pipeline([
    ('pre', preprocessor_default),
    ('clf', LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ))
])

# 6) Train a simple logistic regression model on unbalanced data
pipeline_default.fit(X_train_df, y_train)

# 6.1) 5-Fold Cross-Validation on unbalanced training data
def run_cv(estimator, X, y, name):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
    print(f"\n=== 5-Fold CV ({name}) — macro-F1 scores ===")
    for i, s in enumerate(scores, 1):
        print(f" Fold {i}: {s:.3f}")
    print(f" Average macro-F1: {scores.mean():.3f}\n")

run_cv(pipeline_default, X_train_df, y_train, 'Unbalanced Training Set')

# 7) Check performance on test set
print("\n=== Test Results (UNBALANCED) ===")
test_preds = pipeline_default.predict(X_test_df)
print(classification_report(y_test, test_preds, target_names=["fake","real"]))
cm = confusion_matrix(y_test, test_preds, labels=["fake","real"])
ConfusionMatrixDisplay(cm, display_labels=["fake","real"]).plot(cmap='Blues')
plt.title("Confusion Matrix — Test (Unbalanced)")
plt.show()

# 8) Repeat the exact same pipeline, but load the BALANCED splits instead
train_b = pd.read_csv("train_split_balanced.csv")
test_b  = pd.read_csv("test_split_balanced.csv")

y_train_b = train_b["label"]
y_test_b  = test_b["label"]

# 9) Separate text / metadata from each balanced DataFrame
X_train_b_df = train_b
X_test_b_df  = test_b

# 10) 5-Fold Cross-Validation on balanced training data before hyperparameter tuning
run_cv(pipeline_default, X_train_b_df, y_train_b, 'Balanced Training Set (pre-tuning)')

# 11) Train & eval on balanced data before hyperparameter tuning
print("\n=== Test Results (BALANCED) ===")
test_preds_b = pipeline_default.predict(X_test_b_df)
print(classification_report(y_test_b, test_preds_b, target_names=["fake","real"]))
cm = confusion_matrix(y_test_b, test_preds_b, labels=["fake","real"])
ConfusionMatrixDisplay(cm, display_labels=["fake","real"]).plot(cmap='Blues')
plt.title("Confusion Matrix — Test (Balanced)")
plt.show()

# 12) Hyperparameter tuning for TF-IDF on balanced data using GridSearchCV
param_grid = {
    'pre__text__max_features': [4000, 5000, 6000],
    'pre__text__min_df':       [3, 4, 5],
    'pre__text__max_df':       [0.6, 0.7, 0.8],
    'pre__text__ngram_range':  [(1,1), (1,2), (1,3)]
}
gs = GridSearchCV(  # Use GridSearchCV to find the best hyperparameters
    pipeline_default,
    param_grid,
    scoring='f1_macro',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    error_score='raise'
)
gs.fit(X_train_b_df, y_train_b)
print(f"\n=== Best Hyperparameters (Balanced Training) ===")
print(f" Macro-F1 (CV): {gs.best_score_:.3f}")
print(" Params:", gs.best_params_)

# 13) Nested CV for thresholding and hyperparameter selection
def nested_cv_threshold(pipeline, param_grid, X, y):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores, fold_thresholds = [], []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        # inner CV for hyperparameters
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        gs_inner = GridSearchCV(pipeline, param_grid, scoring='f1_macro', cv=inner_cv, n_jobs=-1)
        gs_inner.fit(X_tr, y_tr)
        best_pipe = gs_inner.best_estimator_

        # threshold selection inside this fold
        probs_cv = cross_val_predict(best_pipe, X_tr, y_tr, cv=inner_cv, method='predict_proba')[:, 1]
        thresholds = np.linspace(0, 1, 101)
        best_thresh = thresholds[np.argmax([
            f1_score(y_tr, np.where(probs_cv >= t, "real", "fake"), average='macro')
            for t in thresholds
        ])]
        fold_thresholds.append(best_thresh)

        # evaluate on outer test fold
        probs_te = best_pipe.predict_proba(X_te)[:, 1]
        preds_te = np.where(probs_te >= best_thresh, "real", "fake")
        score_te = f1_score(y_te, preds_te, average='macro')
        fold_scores.append(score_te)

        print(f"Fold {fold}: macro-F1={score_te:.3f} (threshold={best_thresh:.2f})")

    print(f"\nNested CV macro-F1 avg: {np.mean(fold_scores):.3f}")
    return fold_scores, fold_thresholds

# run nested CV on the balanced training set
scores, thresholds = nested_cv_threshold(pipeline_default, param_grid, X_train_b_df, y_train_b)

# 14) Final Test Results After Threshold Tuning
avg_thresh = np.mean(thresholds)
print(f"\n=== Applying average threshold ({avg_thresh:.2f}) on test set ===")
best_model = gs.best_estimator_
best_model.fit(X_train_b_df, y_train_b)
probs_test_b = best_model.predict_proba(X_test_b_df)[:, 1]
test_preds_final = np.where(probs_test_b >= avg_thresh, "real", "fake")
print(classification_report(y_test_b, test_preds_final, target_names=["fake","real"]))
macro_f1 = f1_score(y_test_b, test_preds_final, average="macro")
print(f"Macro-F1 on test set @ threshold {avg_thresh:.2f}: {macro_f1:.3f}")
cm = confusion_matrix(y_test_b, test_preds_final, labels=["fake","real"])
ConfusionMatrixDisplay(cm, display_labels=["fake","real"]).plot(cmap='Blues')
plt.title(f"Confusion Matrix — Test (Balanced @ threshold={avg_thresh:.2f})")
plt.show()
