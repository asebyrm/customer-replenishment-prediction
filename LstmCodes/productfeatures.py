import pandas as pd
import numpy as np

product_catalog = pd.read_csv('../data/product_catalog.csv')
product_category_map = pd.read_csv('../data/product_category_map.csv')


# Step 1: Explode the categories list in product_catalog
product_catalog['categories'] = product_catalog['categories'].apply(lambda x: eval(x) if pd.notna(x) else [])
exploded_catalog = product_catalog.explode('categories').rename(columns={'categories': 'category_id'})

# Convert category_id to numeric to enable merging
exploded_catalog['category_id'] = pd.to_numeric(exploded_catalog['category_id'], errors='coerce')

# Step 2: Merge with product_category_map to get parent_category_id
merged_catalog = exploded_catalog.merge(product_category_map, on='category_id', how='left')

# Step 3: Aggregate parent_category_id and categories back into lists for each product
final_catalog = merged_catalog.groupby('product_id').agg({
    'manufacturer_id': 'first',
    'attribute_1': 'first',
    'attribute_2': 'first',
    'attribute_3': 'first',
    'attribute_4': 'first',
    'attribute_5': 'first',
    'category_id': lambda x: list(x.dropna()),
    'parent_category_id': lambda x: list(x.dropna())
}).reset_index()

# Rename for clarity
final_catalog = final_catalog.rename(columns={
    'parent_category_id': 'parent_categories',
    'category_id': 'categories'
})


# Save the final catalog with categories and parent_categories
final_catalog.to_csv('../processed_data/final_product_catalog.csv', index=False)

print(final_catalog.head())
