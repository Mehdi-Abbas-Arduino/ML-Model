# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# import geopandas as gpd
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap

# # Load CO2 data
# df = pd.read_csv('cleaned_co2_data.csv')

# # Debugging: Check the loaded DataFrame and its columns
# print(df.head())         # Print the first few rows of the DataFrame
# print(df.columns)        # Print the columns in the DataFrame
# print(df.empty)          # Check if the DataFrame is empty

# # Define features and target variable
# X = df[['decimal_date']]  # Features
# y = df['monthly_average']  # Target variable

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train the model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Evaluate the model
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')

# # Function to predict CO2 levels for a given year
# def predict_co2(year):
#     # Convert year to decimal date (assuming the year is a full year)
#     decimal_date = year + 0.5  # For mid-year prediction
#     new_data = pd.DataFrame({'decimal_date': [decimal_date]})
#     predicted_co2 = model.predict(new_data)
#     return predicted_co2[0]

# # Input year for prediction
# year_input = float(input("Enter a year for CO2 prediction (1958 - 2024): "))
# predicted_co2 = predict_co2(year_input)
# print(f'Predicted CO2 Level for {year_input}: {predicted_co2:.2f}')

# # Load county geometries from GeoJSON
# counties = gpd.read_file("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json")

# # Ensure that the CO2 DataFrame has a 'fips' column for merging
# df['fips'] = df['fips'].astype(str)  # Ensure 'fips' is of type str for merging
# merged = counties.set_index('id').join(df.set_index('fips'))

# # Debugging: Check the merged DataFrame
# print(merged.shape)  # Check the shape of the merged DataFrame
# print(merged.head())  # Print the first few rows to ensure data is present

# # Create a figure and axis for the map
# fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# # Use a custom colormap from light to dark red
# custom_reds = LinearSegmentedColormap.from_list('custom_reds', ['#ffcccc', '#cc0000'])

# # Plot the choropleth map for CO2 levels based on the predicted CO2 value
# merged['monthly_average'] = merged['monthly_average'].fillna(predicted_co2)  # Fill missing values with the predicted CO2 level

# # Check for missing values in the column
# print(merged['monthly_average'].isnull().sum())

# # Use the custom colormap in the plot with vmin and vmax
# merged.plot(column='monthly_average', ax=ax, legend=True,
#             legend_kwds={'label': "CO2 Level (ppm)", 
#                          'orientation': "horizontal"},
#             cmap=custom_reds, missing_kwds={"color": "lightgrey"},
#             vmin=100, vmax=370)  # Adjust the range as needed

# # Add titles and labels
# plt.title(f'US County CO2 Levels for the Year {year_input}')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')

# # Show the predicted CO2 level on the graph (for the specified year)
# plt.annotate(f'Predicted CO2 Level for {year_input}: {predicted_co2:.2f}', 
#              xy=(0.5, 0.9), 
#              xycoords='axes fraction',
#              fontsize=12, 
#              ha='center',
#              bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

# # Show the plot
# plt.show()

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
print(f'Predicted CO2 Level for {year_input}: {predicted_co2:.2f}')

# Create a list to hold predictions for the last 10 years
years = list(range(2014, 2025))  # Adjust this range based on the years you want to predict
predictions = [predict_co2(year) for year in years]

# Plotting the line graph
plt.figure(figsize=(12, 6))
plt.plot(years, predictions, marker='o', linestyle='-', color='red', label='Predicted CO2 Levels')
plt.axhline(y=predicted_co2, color='green', linestyle='--', label=f'Predicted CO2 Level for {year_input}: {predicted_co2:.2f}')

# Add titles and labels
plt.title('Predicted CO2 Levels Over the Years')
plt.xlabel('Year')
plt.ylabel('CO2 Level (ppm)')
plt.xticks(years)  # Set x-ticks to be the years
plt.legend()
plt.grid()

# Show the plot
plt.show()
