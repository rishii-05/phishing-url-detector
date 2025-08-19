# AI-Powered Phishing URL Detector 

An intelligent web application that uses Natural Language Processing (NLP) to detect phishing URLs in real-time. This project demonstrates a complete machine learning workflow, from data cleaning and model training to deployment via a user-friendly Flask interface.

---

## 🚀 The Story Behind The Project: Evolution & Key Learnings

This project began as an implementation of my resume's description: a phishing detector using a **Neural Network** trained on hand-crafted URL attributes (like URL length, character counts, etc.). While this initial model achieved decent accuracy, I quickly discovered its limitations. It was rigid and would often misclassify legitimate modern websites that had structural similarities to phishing links (e.g., flagging a safe URL just because it contained a hyphen).

This led me to evolve the project. I re-engineered the core logic to use a more powerful **Natural Language Processing (NLP)** approach with a Logistic Regression model. Instead of just looking at the URL's structure, this new model treats the URL like a sentence, learning which *words* and *phrases* are most associated with phishing attempts.

This upgrade significantly improved the model's intelligence and accuracy. However, it also revealed a fascinating challenge: **dataset bias**. The model learned to be suspicious of words like "login" or uncommon technical terms like "leet" because they appeared more frequently in phishing URLs within the training data. This highlighted that the true challenge of a real-world AI system is not just building a model, but understanding its limitations and the data that shapes its "worldview".

---

## ✨ Features

-   **Real-Time Analysis:** Instantly classifies any submitted URL as either "Safe" or "Phishing".
-   **NLP-Powered Engine:** Utilizes a Scikit-learn pipeline with `CountVectorizer` to understand the textual content of URLs.
-   **Confidence Score:** Provides a probability score to show how confident the model is in its prediction.
-   **Clean Web Interface:** A simple and intuitive UI built with Flask and HTML/CSS.
-   **Scalable & Modular:** The project is structured to be easily understood, maintained, and updated.

---

## 🛠️ Tech Stack

-   **Backend:** Python, Flask
-   **Machine Learning:** Scikit-learn, Pandas, NLTK
-   **Frontend:** HTML, CSS
-   **Data Source:** [Phishing Site URLs on Kaggle](https://www.kaggle.com/datasets/eshandeorukhkar/phishing-site-urls)

---

## ⚙️ Setup and Local Installation

Follow these steps to get the project running on your local machine.

### 1. Prerequisites

-   Python 3.8 or higher
-   `pip` and `venv`

### 2. Clone the Repository

```
git clone https://github.com/rishii-05/phishing-url-detector.git
cd phishing-url-detector
```

### 3. Set Up a Virtual Environment

It's highly recommended to use a virtual environment to keep dependencies isolated.

```
# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies

Install all the required libraries from the `requirements.txt` file.

```
pip install -r requirements.txt
```

### 5. Train the Model

Before running the web app, you need to train the NLP model. This script will process the data and save the trained pipeline to `app/model/`.

```
python scripts/train_nlp_model.py
```

### 6. Run the Flask Application

Now you can start the web server.

```
python app/app.py
```

Open your web browser and navigate to `http://127.0.0.1:5000` to see the application in action!

---

## 📂 Project Structurephishing_detector/

```
|
├── app/                  # Contains the Flask application
│   ├── static/           # CSS and other static files
│   ├── templates/        # HTML templates
│   ├── model/            # Stores the saved model pipeline
│   └── app.py            # Main Flask application logic
|
├── data/                 # Raw dataset
|
├── scripts/              # Standalone scripts for tasks
│   └── train_nlp_model.py # Script to train the NLP model
|
├── requirements.txt      # Project dependencies
└── README.md             # You are here!
```

---

## 🔮 Future Improvements
- **Retrain with a Better Dataset:** The model's primary limitation is its training data. Retraining on a larger, more balanced dataset that includes more legitimate login pages and tech websites would significantly reduce false positives.
- **Build a Hybrid Model:** The ultimate solution would be a hybrid model that combines this NLP approach with structural and domain-level features (e.g., domain age, SSL certificate validity, WHOIS information) to give the model more context.
