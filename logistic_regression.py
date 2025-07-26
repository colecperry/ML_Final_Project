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

# 1) Read the pre-split files
train_df = pd.read_csv("train_split.csv")
val_df   = pd.read_csv("val_split.csv") 
test_df  = pd.read_csv("test_split.csv")

# 2) Separate features/labels
X_train_df = train_df
y_train    = train_df["label"]

X_val_df   = val_df
y_val      = val_df["label"]

X_test_df  = test_df
y_test     = test_df["label"]

# 3) Build TF-IDF + metadata pipeline
preprocessor_default = ColumnTransformer([
    ('text', TfidfVectorizer(), 'combined_text'),
    ('meta', StandardScaler(), ['publisher_id', 'date_ordinal'])
])
pipeline_default = Pipeline([
    ('pre', preprocessor_default),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

# 4) Train on unbalanced data
pipeline_default.fit(X_train_df, y_train)

# 5) 5-Fold CV on unbalanced training
def run_cv(estimator, X, y, name):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
    print(f"\n=== 5-Fold CV ({name}) — macro-F1 scores ===")
    for i, s in enumerate(scores, 1):
        print(f" Fold {i}: {s:.3f}")
    print(f" Average macro-F1: {scores.mean():.3f}\n")

run_cv(pipeline_default, X_train_df, y_train, 'Unbalanced Training Set')

# 6) Check performance on VALIDATION set (unbalanced)
print("\n=== Validation Results (UNBALANCED) ===")
val_preds = pipeline_default.predict(X_val_df)
print(classification_report(y_val, val_preds, target_names=["fake","real"]))
cm = confusion_matrix(y_val, val_preds, labels=["fake","real"])
ConfusionMatrixDisplay(cm, display_labels=["fake","real"]).plot(cmap='Blues')
plt.title("Confusion Matrix — Validation (Unbalanced)")
plt.show()

# 7) Repeat on balanced splits
train_b = pd.read_csv("train_split_balanced.csv")
val_b   = pd.read_csv("val_split_balanced.csv") 
test_b  = pd.read_csv("test_split_balanced.csv")

X_train_b_df = train_b
y_train_b    = train_b["label"]

X_val_b_df   = val_b
y_val_b      = val_b["label"]

X_test_b_df  = test_b
y_test_b     = test_b["label"]

run_cv(pipeline_default, X_train_b_df, y_train_b, 'Balanced Training Set (pre-tuning)')

print("\n=== Validation Results (BALANCED) ===")
val_preds_b = pipeline_default.predict(X_val_b_df)
print(classification_report(y_val_b, val_preds_b, target_names=["fake","real"]))
cm = confusion_matrix(y_val_b, val_preds_b, labels=["fake","real"])
ConfusionMatrixDisplay(cm, display_labels=["fake","real"]).plot(cmap='Blues')
plt.title("Confusion Matrix — Validation (Balanced)")
plt.show()

# 8) Hyperparameter tuning on balanced training
param_grid = {
    'pre__text__max_features': [4000, 5000, 6000],
    'pre__text__min_df':       [3, 4, 5],
    'pre__text__max_df':       [0.6, 0.7, 0.8],
    'pre__text__ngram_range':  [(1,1), (1,2), (1,3)]
}
gs = GridSearchCV(
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

# 9) Nested CV for threshold + hyperparam selection (uses only train_b)
def nested_cv_threshold(pipeline, param_grid, X, y):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores, fold_thresholds = [], []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        gs_inner = GridSearchCV(pipeline, param_grid, scoring='f1_macro', cv=inner_cv, n_jobs=-1)
        gs_inner.fit(X_tr, y_tr)
        best_pipe = gs_inner.best_estimator_

        probs_cv = cross_val_predict(best_pipe, X_tr, y_tr, cv=inner_cv, method='predict_proba')[:, 1]
        thresholds = np.linspace(0, 1, 101)
        best_thresh = thresholds[np.argmax([
            f1_score(y_tr, np.where(probs_cv >= t, "real", "fake"), average='macro')
            for t in thresholds
        ])]
        fold_thresholds.append(best_thresh)

        probs_te = best_pipe.predict_proba(X_te)[:, 1]
        preds_te = np.where(probs_te >= best_thresh, "real", "fake")
        score_te = f1_score(y_te, preds_te, average='macro')
        fold_scores.append(score_te)

        print(f"Fold {fold}: macro-F1={score_te:.3f} (threshold={best_thresh:.2f})")

    print(f"\nNested CV macro-F1 avg: {np.mean(fold_scores):.3f}")
    return fold_scores, fold_thresholds

scores, thresholds = nested_cv_threshold(pipeline_default, param_grid, X_train_b_df, y_train_b)

# 10) Final Test Results After Threshold Tuning (the only time we touch test_b)
avg_thresh = np.mean(thresholds)
print(f"\n=== Final Test Results @ threshold {avg_thresh:.2f} ===")
best_model = gs.best_estimator_
best_model.fit(X_train_b_df, y_train_b)
probs_test_b = best_model.predict_proba(X_test_b_df)[:, 1]
test_preds_final = np.where(probs_test_b >= avg_thresh, "real", "fake")

print(classification_report(y_test_b, test_preds_final, target_names=["fake","real"]))
print(f"Macro-F1 on test set @ threshold {avg_thresh:.2f}: "
      f"{f1_score(y_test_b, test_preds_final, average='macro'):.3f}")

cm = confusion_matrix(y_test_b, test_preds_final, labels=["fake","real"])
ConfusionMatrixDisplay(cm, display_labels=["fake","real"]).plot(cmap='Blues')
plt.title(f"Confusion Matrix — Test (Balanced @ threshold={avg_thresh:.2f})")
plt.show()
