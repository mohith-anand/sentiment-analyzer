from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Mapping numeric prediction to sentiment label
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    
    # Vectorize and predict
    vect_text = vectorizer.transform([text])
    pred_num = model.predict(vect_text)[0]
    
    # Map to label
    pred_label = label_map.get(pred_num, "Unknown")
    
    return jsonify({'prediction': pred_label})

if __name__ == '__main__':
    app.run(debug=True)
