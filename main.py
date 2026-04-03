import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {len(df)} rows.")
    return df

def clean_data(df):
    print("Cleaning data...")
    # Fill missing values
    df["Message"] = df["Message"].fillna("")
    df["Subject"] = df["Subject"].fillna("")

    # Combine Subject and Message
    df["text"] = df["Subject"] + " " + df["Message"]
    
    # Map labels: ham -> 0, spam -> 1
    df["label"] = df["Spam/Ham"].map({"ham": 0, "spam": 1})

    def clean_text(text):
        text = text.lower()
        # Remove non-word characters and extra spaces
        text = re.sub(r'\W+', ' ', text)
        return text.strip()
    
    df["text"] = df["text"].apply(clean_text)
    
    # Returning raw text and labels for the Pipeline
    return df["text"], df["label"]

def split_data(X, y):
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    print("Training model (TF-IDF + Naive Bayes)...")
    # Using a Pipeline to handle vectorization and classification together
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words='english')),
        ("nb", MultinomialNB())
    ])

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def main():
    print("Welcome to the Spam Detection project!")
    data_path = "data/enron_spam_data.csv"
    
    # 1. Load
    try:
        df = load_data(data_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}")
        return

    # 2. Clean
    X_text, y = clean_data(df)

    # 3. Split
    X_train, X_test, y_train, y_test = split_data(X_text, y)

    # 4. Train
    model = train_model(X_train, y_train)

    # 5. Evaluate
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
