import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Read the pre-split files: take our preprocessed CSV files and load them into DataFrames
train = pd.read_csv("train_split.csv")
val   = pd.read_csv("val_split.csv")
test  = pd.read_csv("test_split.csv")

# 2) Separate text and metadata (publisher_id and date_ordinal) from each DataFrame
texts_train = train["combined_text"]
meta_train  = train[["publisher_id", "date_ordinal"]].values
y_train     = train["label"]

texts_val   = val["combined_text"]
meta_val    = val[["publisher_id", "date_ordinal"]].values
y_val       = val["label"]

texts_test  = test["combined_text"]
meta_test   = test[["publisher_id", "date_ordinal"]].values
y_test      = test["label"]

# 3) Turn text into TF-IDF vectors
tfidf = TfidfVectorizer(max_features=10000)  # Use the 10,000 most common terms
X_text_train = tfidf.fit_transform(texts_train)  # Build vocabulary from training text
X_text_val   = tfidf.transform(texts_val)        # Transform validation text
X_text_test  = tfidf.transform(texts_test)       # Transform test text

# 4) Scale the metadata (publisher_id and date_ordinal)
scaler = StandardScaler()
X_meta_train = scaler.fit_transform(meta_train)  # Fit on training metadata, then transform
X_meta_val   = scaler.transform(meta_val)        # Transform validation metadata
X_meta_test  = scaler.transform(meta_test)       # Transform test metadata

# 5) Combine text features and metadata into one array for each split
X_train = np.hstack([X_text_train.toarray(), X_meta_train])
X_val   = np.hstack([X_text_val.toarray(),   X_meta_val])
X_test  = np.hstack([X_text_test.toarray(),  X_meta_test])

# 6) Train a simple logistic regression model
model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 7) Check performance on validation set
print("=== Validation Results ===")
val_preds = model.predict(X_val)
print(classification_report(y_val, val_preds, target_names=["fake", "real"]))

# Plot confusion matrix for validation set
cm_val = confusion_matrix(y_val, val_preds, labels=["fake", "real"])
disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=["fake", "real"])
plt.figure(figsize=(5, 4))
disp_val.plot(cmap="Blues")
plt.title("Confusion Matrix (Validation Set)")
plt.show()

# 8) Check performance on test set
print("=== Test Results ===")
test_preds = model.predict(X_test)
print(classification_report(y_test, test_preds, target_names=["fake", "real"]))

# Plot confusion matrix for test set
cm_test = confusion_matrix(y_test, test_preds, labels=["fake", "real"])
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=["fake", "real"])
plt.figure(figsize=(5, 4))
disp_test.plot(cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.show()
