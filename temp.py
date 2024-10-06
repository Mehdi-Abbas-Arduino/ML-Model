import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os  # Importing os to check file existence and errors

# Load CO2 data
df = pd.read_csv('cleaned_co2_data.csv')

# Define features and target variable
X = df[['decimal_date']]  # Features
y = df['monthly_average']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model as a pickle file
model_path = 'co2_prediction_model.pkl'

try:
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved successfully at {os.path.abspath(model_path)}")
except Exception as e:
    print(f"Error saving the model: {e}")
