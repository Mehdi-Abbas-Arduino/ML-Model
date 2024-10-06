import pandas as pd

# Load your CO2 data from a CSV file
# Replace 'cleaned_co2_data.csv' with the path to your actual CSV file
df = pd.read_csv('cleaned_co2_data.csv')

# Check the loaded DataFrame and its columns
print("Initial DataFrame:")
print(df.head())         # Print the first few rows of the DataFrame
print("Columns in DataFrame:")
print(df.columns)        # Print the columns in the DataFrame

# Ensure the 'fips' column is of type string in both DataFrames
df['fips'] = df['fips'].astype(str)

# Define a list of counties to add
counties = [
    {'county_name': 'Los Angeles County', 'fips': '06037'},
    {'county_name': 'Cook County', 'fips': '17031'},
    {'county_name': 'Harris County', 'fips': '48201'},
    {'county_name': 'Miami-Dade County', 'fips': '12086'},
    {'county_name': 'King County', 'fips': '53033'},
    {'county_name': 'San Diego County', 'fips': '06073'},
    # Add more counties as needed
]

# Convert the list of counties to a DataFrame
counties_df = pd.DataFrame(counties)

# Ensure the 'fips' column in counties_df is also of type string
counties_df['fips'] = counties_df['fips'].astype(str)

# Merge the county names into the original DataFrame based on the FIPS code
df = df.merge(counties_df, how='left', on='fips')

# Check if the county names were added correctly
print("DataFrame with County Names:")
print(df[['county_name', 'fips']].head())

# Save the updated DataFrame to a new CSV file
df.to_csv('updated_co2_data_with_county_names.csv', index=False)
print("County names have been added and saved to 'updated_co2_data_with_county_names.csv'.")
