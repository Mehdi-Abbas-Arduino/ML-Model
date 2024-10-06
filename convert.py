import pandas as pd

# Replace 'your_file.txt' with the path to your text file
input_file = 'index.txt'
output_file = 'data.csv'

# Read the table from the text file
# Adjust the delimiter as needed (e.g., '\t' for tab, ',' for comma, etc.)
df = pd.read_csv(input_file, delimiter='\t')  # Adjust the delimiter as needed

# Print the first few rows to see the structure of the data``
print(df.head())  # Optional: to see the loaded data and its columns

# Assuming the year is in a column named 'year'
# Adjust 'year' to the actual name of the year column in your DataFrame
year_column_name = 'year'  # Change this to the actual column name containing years

# Check if the year column exists
if year_column_name not in df.columns:
    raise ValueError(f"'{year_column_name}' column not found in the data.")

# Extract unique years
unique_years = df[year_column_name].unique()

# Create a new DataFrame for the unique years
years_df = pd.DataFrame(unique_years, columns=[year_column_name])

# Save the unique years to a CSV file
years_df.to_csv(output_file, index=False)

print(f"Unique years have been saved to {output_file}.")
