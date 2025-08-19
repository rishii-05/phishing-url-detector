from flask import Flask, request, render_template
import pickle
import os

# --- INITIALIZE FLASK APP ---
app = Flask(__name__)

# --- LOAD THE TRAINED PIPELINE ---
# The pipeline includes the vectorizer and the model
base_dir = os.path.abspath(os.path.dirname(__file__))
pipeline_path = os.path.join(base_dir, 'model', 'phishing_pipeline.pkl')

try:
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
    print("Pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading pipeline: {e}")
    pipeline = None

# --- FLASK ROUTES ---
@app.route('/')
def home():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request."""
    if pipeline is None:
        return render_template('index.html', prediction_text='Error: Model not loaded. Please check server logs.')

    try:
        # Get URL from the form
        url_to_check = request.form['url']
        
        # The pipeline expects a list of items to predict, so we wrap the URL in a list
        prediction_array = pipeline.predict([url_to_check])
        prediction_proba_array = pipeline.predict_proba([url_to_check])

        # The result is the first (and only) item in the prediction array
        result_label = prediction_array[0]
        
        # Get the confidence score
        confidence = prediction_proba_array[0].max()

        # Determine the result text
        if result_label == 'bad':
            result = f"This URL is likely a PHISHING site (Confidence: {confidence*100:.2f}%)"
        else:
            result = f"This URL appears to be SAFE (Confidence: {confidence*100:.2f}%)"
            
        return render_template('index.html', prediction_text=result, url_checked=url_to_check)

    except Exception as e:
        error_message = f"An error occurred: {e}"
        return render_template('index.html', prediction_text=error_message, url_checked=request.form.get('url', ''))


# --- RUN THE APP ---
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
