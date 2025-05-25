import numpy as np
import pandas as pd

# Load the datasets
transactions = pd.read_csv('../data/transactions.csv')  # Transactions data containing purchase records
test_data = pd.read_csv('../data/test.csv')  # Test dataset for prediction
product_catalog = pd.read_csv('../data/product_catalog.csv')  # Product catalog
product_category_map = pd.read_csv('../data/product_category_map.csv')  # Product category mapping

# Display the first few rows of the transactions data
transactions.head()

# Convert the purchase date to datetime format
transactions['purchase_date'] = pd.to_datetime(transactions['purchase_date'])

# Sort transactions by customer, purchase date, and product ID
transactions.sort_values(by=["customer_id", "purchase_date", "product_id"], inplace=True)

# Extract the week from the purchase date
transactions['week'] = transactions['purchase_date'].dt.to_period('W')  # Create a weekly period column

# Aggregate weekly purchase quantities for each customer-product combination
weekly_data = transactions.groupby(['customer_id', 'product_id', 'week']).agg({'quantity': 'sum'}).reset_index()

# Pivot the data to create a weekly time series
df_pivot = weekly_data.pivot_table(
    index=["customer_id", "product_id"],  # Use customer and product IDs as the index
    columns="week",  # Use weeks as columns
    values="quantity",  # Fill the table with the quantity values
    fill_value=0  # Fill missing values with 0
).reset_index()

# Convert column names to strings (required for CSV export)
df_pivot.columns = df_pivot.columns.astype(str)

# Ensure customer and product IDs are of type float64
df_pivot["customer_id"] = df_pivot["customer_id"].astype("float64")
df_pivot["product_id"] = df_pivot["product_id"].astype("float64")

# Save the processed data to a CSV file
df_pivot.to_csv("../processed_data/merged_data.csv", index=False)
print(df_pivot.columns)  # Print the column names of the final pivoted DataFrame
