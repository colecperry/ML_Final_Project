import pandas as pd  # for loading and manipulating tables
import re            # for cleaning text via regular expressions
from sklearn.model_selection import train_test_split  # to split data into train/val/test

# ------------------------------
# Step 1: Load FakeNewsNet CSV into a DataFrame
# ------------------------------
df = pd.read_csv('fake.csv')

# ------------------------------
# Step 2: Clean text fields (title + text)
# ------------------------------
def clean_text(text):
    """
    1. If 'text' is not a string (e.g. NaN), return an empty string.
    2. Otherwise, remove HTML tags, URLs, emojis, and unwanted chars, then lowercase.
    """
    if not isinstance(text, str):
        return ""  # Return empty string for non-text inputs or missing values

    # Remove HTML tags, URLs, and non-alphanumeric (keeping basic punctuation)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s,.;!?\'"-]', '', text)
    return text.lower().strip()

print("\nColumns available before cleaning:", df.columns.tolist())

# Apply cleaning
df['clean_title'] = df['title'].apply(clean_text)
df['clean_body']  = df['text'].apply(clean_text)

# Combine into a single 'combined_text' column
df['combined_text'] = df['clean_title'] + ' ' + df['clean_body']

print("\n=== Step 2: After Cleaning ===")
print("Sample original title + cleaned version:")
print(pd.DataFrame({
    'orig_title': df['title'].iloc[:2],
    'clean_title': df['clean_title'].iloc[:2]
}))
print("\nSample combined text (first 2 rows):")
for i in range(2):
    print(f"Row {i} combined_text:\n{df['combined_text'].iloc[i]}\n")

# ------------------------------
# Step 3: Encode metadata (site_url & published) to feed into models
# ------------------------------

# Step 3a: Take each unique site_url and assign a unique integer code
publishers = pd.Categorical(df['site_url']) # Assign a unique code to each unique site_url
df['publisher_id'] = publishers.codes  # Extract the integer codes for each site and add as a new column in dataframe

# Step 3b: Takes a timestamp and converts it into an ordinal integer
df['published_date'] = pd.to_datetime(df['published']) # Parse timestamp into pandas datetime object
df['date_ordinal'] = df['published_date'].apply(lambda d: d.toordinal())
# Convert datetime object to ordinal integer (days since day 1, year 1) for easier numerical processing

print("\n=== Step 3: Metadata Encoding ===")
print("Unique site_url → codes (sample):")
print(df[['site_url', 'publisher_id']].drop_duplicates().iloc[:5])
print("\nPublished → ordinal (sample):")
print(df[['published', 'date_ordinal']].head(3))

# ------------------------------
# Step 4: Create a binary label column by grouping multiple misinfo types into “fake” 
# ------------------------------
misinfo_types = {
    'bs', 'conspiracy', 'hate', 'satire', 'state', 'junksci', 'fake'
}

def map_to_label(article_type):
    """
    If article_type is in our misinfo_types set, label it "fake"; otherwise, "real".
    """
    return "fake" if article_type in misinfo_types else "real"

df["label"] = df["type"].apply(map_to_label)

print("\n=== Step 4: Label Check (Option 2 mapping) ===")
print("Unique values in new 'label':", df['label'].unique())
print("Counts of each label:")
print(df['label'].value_counts())

# ------------------------------
# Step 5: Select columns we need for modeling
# ------------------------------
df_final = df[['combined_text', 'publisher_id', 'date_ordinal', 'label']].copy()

print("\n=== Step 5: Final Columns ===")
print("Columns in df_final:", df_final.columns.tolist())
print("Sample of df_final:")
print(df_final.head(3))

# ------------------------------
# Step 6: Split into train / validation / test sets
# ------------------------------
X = df_final[["combined_text", "publisher_id", "date_ordinal"]]
y = df_final["label"]

# 70% train, 30% temporary (randomly split into val/test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,        # balance "fake" vs "real" in both splits
    random_state=42    # seed the split for reproducibility
)

print("\nAfter 70/30 split:")
print(f"  Training set size: {len(X_train)} examples")
print(f"  Temporary set size (to be split): {len(X_temp)} examples")

# Split the 30% temporary into 50% validation and 50% test (each 15% of total)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print("\nAfter splitting temporary into validation and test:")
print(f"  Validation set size: {len(X_val)} examples")
print(f"  Test set size:       {len(X_test)} examples")

print("\nLabel distribution in training set:")
print(y_train.value_counts(normalize=True))
print("\nLabel distribution in validation set:")
print(y_val.value_counts(normalize=True))
print("\nLabel distribution in test set:")
print(y_test.value_counts(normalize=True))

# Save each split as its own DataFrame and CSV
train_df = pd.concat([X_train, y_train], axis=1)
val_df   = pd.concat([X_val,   y_val],   axis=1)
test_df  = pd.concat([X_test,  y_test],  axis=1)

train_df.to_csv("train_split.csv", index=False)
val_df.to_csv("val_split.csv",     index=False)
test_df.to_csv("test_split.csv",   index=False)

print("\nSaved 'train_split.csv', 'val_split.csv', and 'test_split.csv'.")

# ------------------------------
# Step 7: Address Class Imbalance by undersampling the fake class
# ------------------------------
fake_df = df[df['label'] == 'fake'] # Get only the fake articles
real_df = df[df['label'] == 'real'] # Get only the real articles
fake_sample = fake_df.sample(n=len(real_df), random_state=42) # Randomly sample fake articles to match the number of real articles
df_balanced = pd.concat([real_df, fake_sample]).sample(frac=1, random_state=42).reset_index(drop=True) # Create balanced dataframe and shuffle
print("\n=== Step 7: After Undersampling ===")
print("Label distribution after undersampling:")
print(df_balanced['label'].value_counts(normalize=True))

counts = df_balanced['label'].value_counts()
print(f"Fake articles: {counts['fake']}")
print(f"Real articles: {counts['real']}")

# ------------------------------
# Step 8: Select columns for balanced modeling
# ------------------------------
df_final_bal = df_balanced[['combined_text', 'publisher_id', 'date_ordinal', 'label']].copy() # Select the same columns as before

print("\n=== Step 8: Final Columns (Balanced) ===")
print("Columns in df_final_bal:", df_final_bal.columns.tolist())
print(df_final_bal.head(3))

# ------------------------------
# Step 9: Split balanced data into train/val/test
# ------------------------------
Xb = df_final_bal[['combined_text', 'publisher_id', 'date_ordinal']] # Select features (X)
yb = df_final_bal['label'] # Select labels (y)
Xb_train, Xb_temp, yb_train, yb_temp = train_test_split(
    Xb, yb, test_size=0.30, stratify=yb, random_state=42
) # 70% train, 30% temporary (to be split into val/test)
Xb_val, Xb_test, yb_val, yb_test = train_test_split(
    Xb_temp, yb_temp, test_size=0.50, stratify=yb_temp, random_state=42
) # 50% of temporary for validation, 50% for test (each 15% of total)

print("\nBalanced splits:")
print(f" Training: {len(Xb_train)} | Validation: {len(Xb_val)} | Test: {len(Xb_test)}")
print("Distribution (Balanced Training):")
print(yb_train.value_counts(normalize=True))

# Recombine features and labels into a dataframe for each split (Train, Val, Test)
b_train_df = pd.concat([Xb_train, yb_train], axis=1) 
b_val_df   = pd.concat([Xb_val,   yb_val],   axis=1)
b_test_df  = pd.concat([Xb_test,  yb_test],  axis=1)
b_train_df.to_csv('train_split_balanced.csv', index=False)
b_val_df.to_csv('val_split_balanced.csv',     index=False)
b_test_df.to_csv('test_split_balanced.csv',   index=False)
print("\nSaved balanced splits as 'train_split_balanced.csv', 'val_split_balanced.csv', and 'test_split_balanced.csv'.")
