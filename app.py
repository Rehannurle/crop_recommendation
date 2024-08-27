from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Load the saved model with error handling
model_path = 'D:\crop\recoomender\LogisticRegresion.pkl'
if os.path.exists(model_path):
    try:
        LogReg = pickle.load(open(model_path, 'rb'))
    except (OSError, IOError, pickle.PickleError) as e:
        LogReg = None
        print(f"Error loading model: {e}")
else:
    LogReg = None
    print(f"Model file not found: {model_path}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if LogReg is None:
        return render_template('index.html', prediction_text="Model is not available. Please try again later.")

    try:
        data = request.form.to_dict()
        features = [
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]

        prediction = LogReg.predict([features])
        return render_template('index.html', prediction_text=f'Recommended Crop: {prediction[0]}')

    except ValueError as e:
        return render_template('index.html', prediction_text=f"Input error: {e}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"An error occurred: {e}")

if __name__ == '__main__':
    app.run(debug=True)
