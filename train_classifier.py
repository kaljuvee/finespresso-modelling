
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import spacy
import os
from datetime import datetime

from database import Base, News, Price
from data_manager import DataManager

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model \'en_core_web_sm\'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    if not isinstance(text, str): # Handle non-string input
        return ""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    return " ".join(tokens)

def train_model(model_output_dir="/home/ubuntu/finespresso-modelling/models/"):
    dm = DataManager()
    all_news = dm.get_all_news()

    if not all_news:
        print("No news data found in the database. Cannot train models.")
        print("Creating dummy data for demonstration purposes.")
        dummy_data = {
            'text': [
                "Company A announces positive earnings results",
                "Company B faces regulatory challenges",
                "Company C signs new partnership deal",
                "Company D reports lower than expected profits",
                "Company E clinical trial shows promising results",
                "Company F acquires competitor",
                "Company G issues new shares",
                "Company H management changes announced",
                "Company I product launch successful",
                "Company J annual general meeting held"
            ],
            'event_category': [
                "earnings_releases_and_operating_results",
                "regulatory_filings",
                "partnerships",
                "earnings_releases_and_operating_results",
                "clinical_study",
                "mergers_acquisitions",
                "shares_issue",
                "management_changes",
                "product_services_announcement",
                "annual_general_meeting"
            ],
            'price_direction': [
                'UP', 'DOWN', 'UP', 'DOWN', 'UP', 'UP', 'DOWN', 'DOWN', 'UP', 'UP'
            ]
        }
        df = pd.DataFrame(dummy_data)
    else:
        # Convert news objects to a DataFrame
        data = []
        for news_item in all_news:
            # For now, we'll use a placeholder for price_direction.
            # This will need to be determined based on actual price movements later.
            data.append({
                'text': news_item.title + " " + news_item.summary,
                'event_category': 'all_events', # Placeholder, will need proper categorization
                'price_direction': 'UP' # Placeholder, will need actual price movement
            })
        df = pd.DataFrame(data)

    df['processed_text'] = df['text'].apply(preprocess_text)

    # Train a model for each event category
    event_categories = df['event_category'].unique()
    models = {}
    vectorizers = {}
    results = []

    for event in event_categories:
        print(f"\nTraining model for event category: {event}")
        event_df = df[df['event_category'] == event]

        if len(event_df) < 10: # Minimum samples for training
            print(f"Skipping {event} due to insufficient data ({len(event_df)} samples).")
            continue

        X = event_df['processed_text']
        y = event_df['price_direction']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None)

        vectorizer = TfidfVectorizer(max_features=1000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_vec, y_train)

        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy for {event}: {accuracy:.2f}")
        print(classification_report(y_test, y_pred, zero_division=0))

        models[event] = model
        vectorizers[event] = vectorizer
        results.append({
            'event': event,
            'accuracy': accuracy,
            'num_samples': len(event_df)
        })

        # Save model and vectorizer
        joblib.dump(model, f"{model_output_dir}{event}_model.joblib")
        joblib.dump(vectorizer, f"{model_output_dir}{event}_vectorizer.joblib")

    # Train an 'all_events' fallback model
    print("\nTraining 'all_events' fallback model...")
    X_all = df['processed_text']
    y_all = df['price_direction']

    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all if len(y_all.unique()) > 1 else None)

    all_vectorizer = TfidfVectorizer(max_features=1000)
    X_train_all_vec = all_vectorizer.fit_transform(X_train_all)
    X_test_all_vec = all_vectorizer.transform(X_test_all)

    all_model = RandomForestClassifier(n_estimators=100, random_state=42)
    all_model.fit(X_train_all_vec, y_train_all)

    y_pred_all = all_model.predict(X_test_all_vec)
    all_accuracy = accuracy_score(y_test_all, y_pred_all)

    print(f"Accuracy for all_events: {all_accuracy:.2f}")
    print(classification_report(y_test_all, y_pred_all, zero_division=0))

    models['all_events'] = all_model
    vectorizers['all_events'] = all_vectorizer
    results.append({
        'event': 'all_events',
        'accuracy': all_accuracy,
        'num_samples': len(df)
    })

    joblib.dump(all_model, f"{model_output_dir}all_events_model.joblib")
    joblib.dump(all_vectorizer, f"{model_output_dir}all_events_vectorizer.joblib")

    # Save results to a CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{model_output_dir}model_training_results.csv", index=False)
    print(f"Model training results saved to {model_output_dir}model_training_results.csv")

if __name__ == '__main__':
    # Ensure the data and models directories exist
    os.makedirs("/home/ubuntu/finespresso-modelling/data", exist_ok=True)
    os.makedirs("/home/ubuntu/finespresso-modelling/models", exist_ok=True)
    train_model()


