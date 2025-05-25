import pandas as pd
import ast

# Load datasets
transactions = pd.read_csv('../data/transactions.csv')
product_catalog = pd.read_csv('../data/product_catalog.csv')
product_category_map = pd.read_csv('../data/product_category_map.csv')

# Parse the categories column safely
product_catalog['categories'] = product_catalog['categories'].apply(
    lambda x: ast.literal_eval(x) if pd.notnull(x) else []
)

# Explode the categories column
exploded_catalog = product_catalog.explode('categories')

# Filter and convert categories to integer
exploded_catalog = exploded_catalog[exploded_catalog['categories'].notnull()]
exploded_catalog['categories'] = exploded_catalog['categories'].astype(int)

# Merge with category map to get parent_category_id
merged_catalog = pd.merge(
    exploded_catalog,
    product_category_map,
    left_on='categories',
    right_on='category_id',
    how='left'
)

# Aggregate parent categories by product_id
parent_categories_per_product = merged_catalog.groupby('product_id')['parent_category_id'].apply(list).reset_index()

# Debug: Check parent categories
print(parent_categories_per_product.head())

# Merge back to product_catalog
product_catalog = pd.merge(
    product_catalog,
    parent_categories_per_product,
    on='product_id',
    how='left'
)

# Debug: Ensure parent_category_id exists
if 'parent_category_id' not in product_catalog.columns:
    raise ValueError("The 'parent_category_id' column was not properly added to 'product_catalog'.")

# Merge transactions with product_catalog to get parent_category_id
merged_transactions = pd.merge(
    transactions,
    product_catalog[['product_id', 'parent_category_id']],
    on='product_id',
    how='left'
)

# Explode parent_category_id to handle multiple categories
merged_transactions = merged_transactions.explode('parent_category_id')

# Filter rows with valid parent_category_id
merged_transactions = merged_transactions[merged_transactions['parent_category_id'].notnull()]
merged_transactions['parent_category_id'] = merged_transactions['parent_category_id'].astype(int)

# Group by customer_id and parent_category_id
customer_category_purchases = merged_transactions.groupby(
    ['customer_id', 'parent_category_id']
)['quantity'].sum().reset_index()

# Get top 5 parent categories for each customer
top_categories_per_customer = customer_category_purchases.groupby('customer_id').apply(
    lambda x: x.nlargest(5, 'quantity')
).reset_index(drop=True)

# Combine top categories into a list per customer
top_categories_summary = top_categories_per_customer.groupby('customer_id')['parent_category_id'].apply(list).reset_index()

# Customer-level feature extraction
customer_features = transactions.groupby('customer_id').agg(
    total_purchases=('quantity', 'sum'),
    unique_products=('product_id', 'nunique'),
    purchase_days=('purchase_date', 'nunique'),
    avg_quantity=('quantity', 'mean'),
    frequent_product=('product_id', lambda x: x.value_counts().idxmax())
).reset_index()

# Merge top categories with customer features
customer_features = pd.merge(
    customer_features,
    top_categories_summary,
    on='customer_id',
    how='left'
)

# Save to CSV
customer_features.to_csv('../processed_data/customer_features.csv', index=False)

# Debug output
print(customer_features.head(10))
