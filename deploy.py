import pickle
import pandas as pd
from flask import Flask, request, render_template
from flask_cors import CORS
# Create a Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model from the pickle file
with open('co2_prediction_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the home route
@app.route('/')
def home():
    return render_template('model.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the year input from the form
        year = float(request.form['year'])
        # Convert year to decimal date for prediction
        decimal_date = year + 0.5
        new_data = pd.DataFrame({'decimal_date': [decimal_date]})
        
        # Make the prediction
        prediction = model.predict(new_data)
        
        # Return the prediction in the rendered HTML
        return render_template('model.html', pred=f'Predicted CO2 Level for {year}: {prediction[0]:.2f} ppm')
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return "An error occurred!", 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
