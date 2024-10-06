import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data from a CSV file
# Replace 'co2_data.csv' with the path to your CSV file
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

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (Millions): {mse}')

# Predict CO2 levels for a specific decimal date
new_data = pd.DataFrame({'decimal_date': [1965.0]})  # Replace with your input
predicted_co2 = model.predict(new_data)
print(f'Predicted CO2 Level: {predicted_co2[0]}')
