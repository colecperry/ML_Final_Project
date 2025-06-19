import pandas as pd  # for loading and manipulating tables
import re            # for cleaning text via regular expressions
from sklearn.model_selection import train_test_split  # to split data into train/test

# ------------------------------
# Step 1: Load FakeNewsNet CSV into a DataFrame
# ------------------------------
df = pd.read_csv('fake.csv')

# ------------------------------
# Step 2: Clean text fields (title + text)
# ------------------------------
def clean_text(text):
    """
    1. If text is missing or not a string, return empty string.
    2. Remove HTML tags, URLs, emojis, unwanted characters, then lowercase.
    """
    if not isinstance(text, str):
        return ""  # Handle NaN or non-string
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s,.;!?\'"-]', '', text)
    return text.lower().strip()

print("\nColumns available before cleaning:", df.columns.tolist())

# Apply cleaning
df['clean_title'] = df['title'].apply(clean_text)
df['clean_body']  = df['text'].apply(clean_text)
# Combine into a single 'combined_text'
df['combined_text'] = df['clean_title'] + ' ' + df['clean_body']

print("\n=== Step 2: After Cleaning ===")
print("Sample original title + cleaned version:")
print(pd.DataFrame({'orig_title': df['title'].iloc[:2], 'clean_title': df['clean_title'].iloc[:2]}))
print("\nSample combined text (first 2 rows):")
for i in range(2):
    print(f"Row {i} combined_text:\n{df['combined_text'].iloc[i]}\n")

# ------------------------------
# Step 3: Encode metadata (site_url & published)
# ------------------------------
# 3a: Assign unique integer code to each site_url
df['publisher_id'] = pd.Categorical(df['site_url']).codes

# 3b: Parse timestamps robustly and convert to ordinal integers
df['published_date'] = pd.to_datetime(df['published'], errors='coerce', utc=True)
# Remove timezone info
df['published_date'] = df['published_date'].dt.tz_convert(None)
# Convert to ordinal days
df['date_ordinal'] = df['published_date'].dt.date.apply(lambda d: d.toordinal() if pd.notnull(d) else None)

print("\n=== Step 3: Metadata Encoding ===")
print(df[['site_url', 'publisher_id']].drop_duplicates().iloc[:5])
print(df[['published', 'date_ordinal']].head(3))

# ------------------------------
# Step 4: Create a binary label column grouping misinformation types as “fake”
# ------------------------------
misinfo_types = {'bs', 'conspiracy', 'hate', 'satire', 'state', 'junksci', 'fake'}
df['label'] = df['type'].apply(lambda t: 'fake' if t in misinfo_types else 'real')
print("\n=== Step 4: Label Check ===")
print(df['label'].value_counts())

# ------------------------------
# Step 5: Select columns for modeling
# ------------------------------
df_final = df[['combined_text', 'publisher_id', 'date_ordinal', 'label']].copy()
print("\n=== Step 5: Final Columns ===")
print(df_final.head(3))

# ------------------------------
# Step 6: Balance classes by undersampling fake
# ------------------------------
fake_df = df_final[df_final['label']=='fake']
real_df = df_final[df_final['label']=='real']
fake_sample = fake_df.sample(n=len(real_df), random_state=42)
df_balanced = pd.concat([real_df, fake_sample], ignore_index=True).sample(frac=1, random_state=42)
print("\n=== Step 6: After Undersampling ===")
print(df_balanced['label'].value_counts(normalize=True))

# ------------------------------
# Step 7: Select balanced columns
# ------------------------------
df_final_bal = df_balanced[['combined_text', 'publisher_id', 'date_ordinal', 'label']].copy()
print("\n=== Step 7: Final Columns (Balanced) ===")
print(df_final_bal.head(3))

# ------------------------------
# Step 8: Split balanced data into train and test only (70/30)
# ------------------------------
Xb = df_final_bal[['combined_text', 'publisher_id', 'date_ordinal']]
yb = df_final_bal['label']
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    Xb, yb,
    test_size=0.30,
    stratify=yb,
    random_state=42
)
# Save splits
train_orig = pd.concat([Xb_train, yb_train], axis=1)
test_orig  = pd.concat([Xb_test,  yb_test],  axis=1)
train_orig.to_csv('train_split_balanced.csv', index=False)
test_orig.to_csv('test_split_balanced.csv',  index=False)
print(f"Train: {len(Xb_train)}  |  Test: {len(Xb_test)}")

# ------------------------------
# Step 9: Create separate train/val/test splits for RoBERTa workflow (70/15/15)
# ------------------------------
# Use the original train set
bert_df = train_orig.copy()
X_bert = bert_df[['combined_text','publisher_id','date_ordinal']]
y_bert = bert_df['label']

# 15% of overall data ⇒ ~0.214 of bert_df
X_bert_train, X_bert_val, y_bert_train, y_bert_val = train_test_split(
    X_bert, y_bert,
    test_size=0.214,
    stratify=y_bert,
    random_state=42
)

# Save BERT-specific splits
pd.concat([X_bert_train, y_bert_train], axis=1).to_csv('train_split_bert.csv', index=False)
pd.concat([X_bert_val,   y_bert_val],   axis=1).to_csv('val_split_bert.csv',   index=False)
pd.concat([Xb_test,      yb_test],      axis=1).to_csv('test_split_bert.csv', index=False)
print(f"BERT splits – Train: {len(X_bert_train)}, Val: {len(X_bert_val)}, Test: {len(Xb_test)}")
