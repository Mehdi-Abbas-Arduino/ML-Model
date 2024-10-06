import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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

# Function to predict CO2 levels for a given year
def predict_co2(year):
    # Convert year to decimal date (assuming the year is a full year)
    decimal_date = year + 0.5  # For mid-year prediction
    new_data = pd.DataFrame({'decimal_date': [decimal_date]})
    predicted_co2 = model.predict(new_data)
    return predicted_co2[0]

# Input year for prediction
year_input = float(input("Enter a year for CO2 prediction (1958 - 2024): "))
predicted_co2 = predict_co2(year_input)
predicted_co2_next = predict_co2(year_input + 1)  # Predict CO2 level for the next year
print(f'Predicted CO2 Level for {year_input}: {predicted_co2:.2f}')
print(f'Predicted CO2 Level for {year_input + 1}: {predicted_co2_next:.2f}')

# Create a list to hold predictions for the last 5 years including the input year and extend to 2025
years = list(range(int(year_input) - 4, 2026))  # 5 years before and include up to 2025
predictions = [predict_co2(year) for year in years]

# Plotting the line graph
plt.figure(figsize=(12, 6))
plt.plot(years, predictions, marker='o', linestyle='-', color='red', label='Predicted CO2 Levels')
plt.axhline(y=predicted_co2, color='green', linestyle='--', label=f'Predicted CO2 Level for {year_input}: {predicted_co2:.2f}')
plt.axhline(y=predicted_co2_next, color='blue', linestyle='--', label=f'Predicted CO2 Level for {year_input + 1}: {predicted_co2_next:.2f}')

# Add titles and labels
plt.title('Predicted CO2 Levels Over the Years')
plt.xlabel('Year')
plt.ylabel('CO2 Level (ppm)')
plt.xticks(years)  # Set x-ticks to be the years
plt.legend()
plt.grid()

# Show the plot
plt.show()
