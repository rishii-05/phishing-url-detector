import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from nltk.tokenize import RegexpTokenizer
import pickle
import os

def main():
    print("Starting NLP model training...")

    # --- 1. LOAD DATA ---
    input_csv_path = os.path.join('data', 'phishing_site_urls.csv')
    
    if not os.path.exists(input_csv_path):
        print(f"Error: Input data file not found at {input_csv_path}")
        return

    df = pd.read_csv(input_csv_path)
    # Drop duplicates and any missing values
    df = df.drop_duplicates()
    df = df.dropna()
    print(f"Loaded and cleaned {len(df)} URLs.")

    # Separate features (URLs) and labels
    urls = df['URL']
    labels = df['Label']

    # --- 2. SPLIT DATA ---
    urls_train, urls_test, labels_train, labels_test = train_test_split(
        urls, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print("Data split into training and testing sets.")

    # --- 3. CREATE THE NLP PIPELINE ---
    # A tokenizer that only considers sequences of letters (ignores numbers, punctuation)
    tokenizer = RegexpTokenizer(r'[A-Za-z]+')

    # A CountVectorizer that uses our tokenizer
    vectorizer = CountVectorizer(
        tokenizer=tokenizer.tokenize,
        stop_words='english', # Removes common English words
        lowercase=True
    )

    # The classification model
    model = LogisticRegression(max_iter=1000) # Increased max_iter for convergence

    # Create the full pipeline:
    # 1. Vectorize the URL text
    # 2. Train the Logistic Regression model
    pipeline = make_pipeline(vectorizer, model)
    print("NLP pipeline created.")

    # --- 4. TRAIN THE MODEL ---
    print("Training the pipeline...")
    pipeline.fit(urls_train, labels_train)
    print("Training complete.")

    # --- 5. EVALUATE THE MODEL ---
    accuracy = pipeline.score(urls_test, labels_test)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")

    # --- 6. SAVE THE PIPELINE ---
    # The pipeline object contains the vectorizer and the trained model.
    pipeline_save_path = os.path.join('app', 'model', 'phishing_pipeline.pkl')
    with open(pipeline_save_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"Pipeline saved to {pipeline_save_path}")

if __name__ == "__main__":
    main()
