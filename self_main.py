import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class MyMultinomialNB:
    """
    Custom Implementation of Multinomial Naive Bayes.
    This class handles text classification based on discrete word counts (TF-IDF values).
    """
    def __init__(self, alpha=1.0):
        """
        Initialize the model.
        :param alpha: Laplace smoothing parameter (prevents zero division and log(0) errors)
        """
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = {}
        self.feature_log_prob_ = {}

    def fit(self, X, y):
        """
        Train the model by calculating priors and feature likelihoods.
        :param X: Feature matrix (TF-IDF weighted)
        :param y: Labels (0 for ham, 1 for spam)
        """
        y = np.array(y)
        self.classes_ = np.unique(y)
        n_samples = X.shape[0]

        for c in self.classes_:
            # Select samples belonging to class 'c'
            X_c = X[y == c]

            # Prior: log P(class) = log(samples in class / total samples)
            # Using log-scale to avoid numerical underflow.
            self.class_log_prior_[c] = np.log(X_c.shape[0] / n_samples)

            # Sum feature values (TF-IDF scores) for all samples in this class.
            # This represents the cumulative "importance" of each word in this class.
            feature_count = np.asarray(X_c.sum(axis=0)).ravel()

            # Laplace smoothing: Add 'alpha' to each word's count.
            # This ensures that even if a word was never seen in a class during training,
            # it still has a non-zero probability.
            smoothed_feature_count = feature_count + self.alpha
            smoothed_feature_total = smoothed_feature_count.sum()

            # Conditional Probability (Likelihood): log P(word | class).
            # log(word's weight / total weight of all words in class)
            self.feature_log_prob_[c] = np.log(
                smoothed_feature_count / smoothed_feature_total
            )

        return self

    def predict(self, X):
        """
        Predict labels for a set of samples.
        """
        predictions = []

        for i in range(X.shape[0]):
            x = X[i]
            class_scores = []

            for c in self.classes_:
                # Posterior Probability calculation in log-scale:
                # log P(class | words) ∝ log P(class) + Σ log P(word_i | class) * weight_i
                log_prior = self.class_log_prior_[c]
                
                # Dot product calculates the weighted sum of log-likelihoods for words in the email.
                log_likelihood = x.dot(self.feature_log_prob_[c])[0]
                
                score = log_prior + log_likelihood
                class_scores.append(score)

            # Assign the class with the highest log-probability score
            predicted_class = self.classes_[np.argmax(class_scores)]
            predictions.append(predicted_class)

        return np.array(predictions)


def load_data(file_path):
    """Loads CSV data using pandas."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {len(df)} rows.")
    return df


def clean_text(text):
    """Performs basic text preprocessing: lowercase, remove punctuation, remove extra space."""
    text = str(text).lower()
    text = re.sub(r"\W+", " ", text) # Remove non-alphanumeric
    text = re.sub(r"\s+", " ", text) # Replace multiple spaces with single space
    return text.strip()


def clean_data(df):
    """Prepares the dataframe for modeling."""
    print("Cleaning data...")

    # Fill empty Subject/Message fields
    df["Message"] = df["Message"].fillna("")
    df["Subject"] = df["Subject"].fillna("")

    # Combine subject and message into a single 'text' column
    df["text"] = df["Subject"] + " " + df["Message"]
    df["text"] = df["text"].apply(clean_text)

    # Encode target labels: ham=0, spam=1
    df["label"] = df["Spam/Ham"].map({"ham": 0, "spam": 1})

    # Remove any rows with missing labels and ensure integer type
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    print("Data cleaning completed.")
    print("Class distribution:")
    print(df["label"].value_counts())

    return df["text"], df["label"]


def split_data(X, y):
    """Splits the text data into training and testing sets."""
    print("Splitting data into train and test sets...")

    # stratify=y ensures both sets have roughly the same spam/ham ratio as the original.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def create_custom_model(X_train, y_train):
    """Applies TF-IDF vectorization and trains the custom Naive Bayes classifier."""
    print("Creating TF-IDF features...")

    # Max features capped at 5000 to keep the feature space manageable.
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    X_train_vec = vectorizer.fit_transform(X_train)

    print("Training custom Multinomial Naive Bayes model...")
    model = MyMultinomialNB(alpha=1.0)
    model.fit(X_train_vec, y_train)

    return vectorizer, model


def evaluate_model(vectorizer, model, X_test, y_test):
    """Evaluates the trained model on the test set and prints metrics."""
    print("Evaluating model...")

    # Transform test strings into vectors using the training-time vectorizer vocabulary.
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    print("\nAccuracy Score:")
    print(accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def predict_new_email(vectorizer, model, subject, message):
    """Inference function for predicting the category of a raw subject and message."""
    text = clean_text(subject + " " + message)
    text_vec = vectorizer.transform([text])
    pred = model.predict(text_vec)[0]
    return "spam" if pred == 1 else "ham"


def main():
    print("--- Spam Detection Script Starting ---")
    data_path = "data/enron_spam_data.csv"

    # 1. Load Dataset
    try:
        df = load_data(data_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}. Please check the file path.")
        return
    except Exception as e:
        print(f"Error while loading dataset: {e}")
        return

    # 2. Clean Data (Pre-processing)
    X_text, y = clean_data(df)

    # 3. Create Training and Testing Sets
    X_train, X_test, y_train, y_test = split_data(X_text, y)

    # 4. Feature Extraction & Model Training
    vectorizer, model = create_custom_model(X_train, y_train)

    # 5. Performance Validation
    evaluate_model(vectorizer, model, X_test, y_test)

    # 6. Real-world Sample Testing
    print("\n--- Manual Predictions ---")
    
    # Sample Test 1: Typical Spam
    subject_1 = "Win a free vacation now"
    msg_1 = "Click here to claim your prize and free money. Limited time offer!"
    print(f"Email 1 Subject: '{subject_1}'")
    print(f"Prediction: {predict_new_email(vectorizer, model, subject_1, msg_1)}")

    # Sample Test 2: Typical Ham (Legitimate)
    subject_2 = "Meeting schedule for tomorrow"
    msg_2 = "Please find attached the agenda for tomorrow's team meeting. See you at 10 AM."
    print(f"\nEmail 2 Subject: '{subject_2}'")
    print(f"Prediction: {predict_new_email(vectorizer, model, subject_2, msg_2)}")


if __name__ == "__main__":
    main()