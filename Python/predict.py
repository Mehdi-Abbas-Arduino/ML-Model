import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Change directory to where your Python files are
os.chdir('Python/')

# Load CO2 data
df = pd.read_csv('cleaned_co2_data.csv')

# Debugging: Check the loaded DataFrame and its columns
print(df.head())         # Print the first few rows of the DataFrame
print(df.columns)        # Print the columns in the DataFrame
print(df.empty)          # Check if the DataFrame is empty

# Define features and target variable
X = df[['decimal_date']]  # Features
y = df['monthly_average']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the trained model to a .pkl file
with open('co2_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model has been saved as co2_model.pkl")

# Function to predict CO2 levels for a given year
def predict_co2(year):
    # Convert year to decimal date (assuming the year is a full year)
    decimal_date = year + 0.5  # For mid-year prediction
    new_data = pd.DataFrame({'decimal_date': [decimal_date]})
    predicted_co2 = model.predict(new_data)
    return predicted_co2[0]

# Function to create and save the CO2 graph
def create_co2_graph(predicted_co2, year_input):
    years = list(range(int(year_input) - 4, int(year_input+5)))  # 5 years before and include up to 2025
    predictions = [predict_co2(year) for year in years]

    plt.figure(figsize=(12, 6))
    plt.plot(years, predictions, marker='o', linestyle='-', color='red', label='Predicted CO2 Levels')
    plt.axhline(y=predicted_co2, color='green', linestyle='--', label=f'Predicted CO2 Level for {year_input}: {predicted_co2:.2f}')
    plt.axhline(y=predict_co2(year_input + 1), color='blue', linestyle='--', label=f'Predicted CO2 Level for {year_input + 1}: {predict_co2(year_input + 1):.2f}')

    # Add titles and labels
    plt.title('Predicted CO2 Levels Over the Years')
    plt.xlabel('Year')
    plt.ylabel('CO2 Level (ppm)')
    plt.xticks(years)  # Set x-ticks to be the years
    plt.legend()
    plt.grid()

    # Save the plot as an image
    plt.savefig('co2_plot.png')  # Save in the static directory
    plt.close()  # Close the plot to free memory



from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This allows all CORS requests

@app.route('/')
def index():
    return render_template('index.html')  # Ensure your HTML file is named index.html or update this accordingly

@app.route("/predict", methods=["POST"])
def predict():
    year_input = request.form.get("Year")  # Get the year from the POST request
    if year_input:
        year_input = int(year_input)  # Convert to integer
        predicted_co2 = predict_co2(year_input)  # Call the prediction function
        create_co2_graph(predicted_co2, year_input)  # Generate the graph
        return jsonify({
            "predicted_co2": predicted_co2,
            "message": f"Graph created for the year {year_input}.",
            "graph_url": "co2_plot.png"
        }), 200  # Return success response with graph URL
    else:
        return jsonify({"error": "Year not provided."}), 400  # Return error response if no year is provided


if __name__ == '__main__':
    app.run(port=5050)
