import pickle
from flask import Flask, request, jsonify

# Load the trained model and DictVectorizer
model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Initialize the Flask application
app = Flask('lead_conversion')

@app.route('/predict', methods=['POST'])
def predict():
    # Parse incoming JSON data
    lead = request.get_json()

    # Transform the lead data using the DictVectorizer
    X = dv.transform([lead])
    
    # Predict conversion probability
    y_pred = model.predict_proba(X)[0, 1]
    converted = y_pred >= 0.5

    # Prepare and return the result
    result = {
        'conversion_probability': float(y_pred),
        'converted': bool(converted)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
