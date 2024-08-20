from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load the preprocessor and model
preprocessor = joblib.load('./models/preprocessor.pkl')
model = joblib.load('./models/xgb_churn_model.pkl')

# Initialize Flask app
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.json

    # Convert JSON to DataFrame
    df = pd.DataFrame([data], index=None)

    # Apply preprocessing pipeline
    processed_data = preprocessor.fit_transform(df)

    # Make predictions
    predictions = model.predict(processed_data)

    # Convert predictions to a list and return as JSON
    return jsonify({'predictions': predictions.tolist()})


if __name__ == '__main__':
    app.run(debug=True)
