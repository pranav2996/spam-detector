import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# =========================
# LOAD DATASET
# =========================
data = pd.read_csv("spam.csv", encoding='latin-1')

# =========================
# HANDLE DIFFERENT DATASETS
# =========================

print("Columns in dataset:", data.columns)

# Case 1: Kaggle dataset (v1, v2)
if 'v1' in data.columns and 'v2' in data.columns:
    data = data[['v1', 'v2']]
    data.columns = ['Category', 'Message']

# Case 2: Already correct dataset
elif 'Category' in data.columns and 'Message' in data.columns:
    data = data[['Category', 'Message']]

# If neither matches → error
else:
    raise Exception("Dataset format not recognized. Check column names.")

# =========================
# CLEAN DATASET
# =========================
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

# Clean labels
data['Category'] = data['Category'].astype(str).str.strip().str.lower()
data['Category'] = data['Category'].map({'ham': 0, 'spam': 1})

# Remove invalid rows
data = data.dropna()

# =========================
# FEATURES & LABELS
# =========================
X = data['Message']
y = data['Category'].astype(int)

print("Unique labels:", y.unique())

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TF-IDF
# =========================
vectorizer = TfidfVectorizer(stop_words='english')
X_train_features = vectorizer.fit_transform(X_train)

# =========================
# MODEL
# =========================
model = MultinomialNB()
model.fit(X_train_features, y_train)

# =========================
# ACCURACY
# =========================
X_test_features = vectorizer.transform(X_test)
accuracy = model.score(X_test_features, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# =========================
# SAVE
# =========================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained and saved successfully!")