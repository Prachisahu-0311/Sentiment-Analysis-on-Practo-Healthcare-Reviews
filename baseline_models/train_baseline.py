import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# PATHS (Updated for your SSH Linux environment)
# ==========================================
# Gets the root directory (my_project_m24mac005) assuming you run this from the root
ROOT_DIR = os.getcwd() 
BASE_DIR = os.path.join(ROOT_DIR, "Doctor_reviews_scraped_text_data", "Practo_scraped_text_data", "output_processed_data")

TRAIN_PATH = os.path.join(BASE_DIR, "model_train_balanced.csv")
TEST_PATH = os.path.join(BASE_DIR, "model_test_real_world.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models") # Saves inside baseline_models/saved_models

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Reading data from: {BASE_DIR}")
print("1. Loading Data...")
train_df = pd.read_csv(TRAIN_PATH).dropna(subset=['review_text_clean'])
test_df = pd.read_csv(TEST_PATH).dropna(subset=['review_text_clean'])

X_train = train_df['review_text_clean']
X_test = test_df['review_text_clean']

# XGBoost requires numerical labels (0 for negative, 1 for positive)
y_train_num = train_df['sentiment_label'].map({'negative': 0, 'positive': 1})
y_test_num = test_df['sentiment_label'].map({'negative': 0, 'positive': 1})

# Logistic Regression uses string labels directly
y_train_str = train_df['sentiment_label']
y_test_str = test_df['sentiment_label']

print("2. TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("3. Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train_str)
lr_preds = lr_model.predict(X_test_vec)

print("4. Training XGBoost...")
xgb_model = XGBClassifier()
xgb_model.fit(X_train_vec, y_train_num)
xgb_preds = xgb_model.predict(X_test_vec)

print("\n=== LOGISTIC REGRESSION EVALUATION (Real-World Test Set) ===")
print(f"Accuracy: {accuracy_score(y_test_str, lr_preds):.4f}")
print(classification_report(y_test_str, lr_preds))

print("\n=== XGBOOST EVALUATION (Real-World Test Set) ===")
xgb_preds_str = ['positive' if p == 1 else 'negative' for p in xgb_preds]
print(f"Accuracy: {accuracy_score(y_test_str, xgb_preds_str):.4f}")
print(classification_report(y_test_str, xgb_preds_str))

print("\n5. Saving Best Baseline Model & Vectorizer...")
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
joblib.dump(lr_model, os.path.join(MODEL_DIR, "baseline_lr_model.pkl"))

print(f"✅ Models saved successfully in {MODEL_DIR}")