import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Embedding, Flatten, LSTM, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Lambda
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Set the matplotlib backend for GUI support


# Parameters for sequence length and dataset size
n_steps = 8  # Number of historical steps in the input
n_future = 4  # Number of future steps to predict
size = 0.01  # Fraction of data to use


# Function to create sequences for model input
def create_sequences(data, catalog, customer_catalog, steps, future):
    """
    Generates input sequences, targets, and additional features for customers and products.

    Parameters:
        data: DataFrame containing the time series and metadata
        catalog: DataFrame with product metadata
        customer_catalog: DataFrame with customer metadata
        steps: Number of pastime steps to include in the input
        future: Number of future steps to predict

    Returns:
        Tuple of arrays for time series input, target values, customer features, and product features.
    """
    X, y, customers_list, products = [], [], [], []

    # Drop unnecessary columns
    customer_catalog.drop(columns=["frequent_product", "parent_category_id"], inplace=True)
    customer_features_dict = customer_catalog.set_index('customer_id').to_dict(orient='index')
    catalog.drop(columns=["categories"], inplace=True)
    product_features_dict = catalog.set_index('product_id').to_dict(orient='index')

    week_columns = [col for col in data.columns if '/' in col]  # Identify weekly data columns

    # Generate sequences for each customer-product pair
    for index in data.index:
        weeks = data.loc[index, week_columns].values
        customer_id = data.loc[index, "customer_id"]
        customer_features = customer_features_dict[customer_id]
        product_id = data.loc[index, "product_id"]
        product_features = product_features_dict[product_id]

        for i in range(len(weeks) - steps):
            seq_x = weeks[i:i + steps].reshape(-1, 1)  # Historical sequence
            future_weeks = weeks[i + steps:i + steps + future]  # Future target weeks
            target_week = calculate_target_multiclass(future_weeks)  # Calculate target label
            X.append(seq_x)
            y.append(target_week)
            customers_list.append([customer_id] + list(customer_features.values()))
            products.append([product_id] + list(product_features.values()))

    return np.array(X), np.array(y), np.array(customers_list), np.array(products)

# Function to calculate multiclass target labels


def calculate_target_multiclass(future_weeks):
    """
    Determines the target class based on the first future week with a sale.

    Parameters:
        future_weeks: Array of future sales data

    Returns:
        The index of the first non-zero week as the target class, or 0 if no sale occurs.
    """
    if np.any(future_weeks):  # Check if there are any future sales
        return np.argmax(future_weeks > 0) + 1  # Return the first week with a sale
    return 0  # No sale in future weeks

# Function to scale and reshape data


def scale_and_reshape(scaler, x, num_steps, num_features):
    """
    Scales and reshapes the input data for the model.

    Parameters:
        scaler: Fitted MinMaxScaler
        x: Input data array
        num_steps: Number of time steps
        num_features: Number of features

    Returns:
        Scaled and reshaped array
    """
    X_reshaped = x.reshape(-1, num_features)
    X_scaled = scaler.transform(X_reshaped).reshape(-1, num_steps, num_features)
    return X_scaled

# Function to calculate embedding output dimensions


def calculate_output_dim(input_dim):
    """
    Determines the optimal output dimension for embeddings.

    Parameters:
        input_dim: Size of the input dimension

    Returns:
        Nearest power of two greater than or equal to the square root of the input dimension.
    """
    sqrt_dim = int(np.sqrt(input_dim))
    nearest_power_of_two = 2**int(np.ceil(np.log2(sqrt_dim)))
    return nearest_power_of_two


# Function to split data into training, validation, and test sets
def split_data(X, y_labels, customers, products, train_ratio=0.7, val_ratio=0.2):
    """
    Splits the input data into training, validation, and test sets, and applies MinMax scaling.

    Parameters:
        X: Feature data
        y_labels: Target data for multiclass classification
        customers: Customer data
        products: Product data
        train_ratio: Proportion of training data
        val_ratio: Proportion of validation data

    Returns:
        Dictionary containing split and scaled data
    """
    total_length = len(X)
    train_size = int(train_ratio * total_length)
    val_size = int(val_ratio * total_length)

    # Split features and targets
    X_train = X[:train_size]
    y_train_multiclass = y_labels[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val_multiclass = y_labels[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test_multiclass = y_labels[train_size + val_size:]

    # Split customers and products
    customers_train = customers[:train_size]
    customers_val = customers[train_size:train_size + val_size]
    customers_test = customers[train_size + val_size:]

    products_train = products[:train_size]
    products_val = products[train_size:train_size + val_size]
    products_test = products[train_size + val_size:]

    # Initialize and fit the scaler
    scaler = MinMaxScaler()
    scaler.fit(X_train.reshape(-1, 1))  # Reshape for scaler compatibility
    joblib.dump(scaler, "../scalers/scaler.pkl")  # Save the scaler for future use
    print("Scaler SAVED at:", "scalers/scaler.pkl")

    # Scale and reshape the splits
    num_steps, num_features = X_train.shape[1], 1  # Assuming X is a 2D array
    X_train_scaled = scale_and_reshape(scaler, X_train, num_steps, num_features)
    X_val_scaled = scale_and_reshape(scaler, X_val, num_steps, num_features)
    X_test_scaled = scale_and_reshape(scaler, X_test, num_steps, num_features)

    # Return the splits as a dictionary
    return {
        "X_train": X_train_scaled, "y_train_multiclass": y_train_multiclass,
        "X_val": X_val_scaled, "y_val_multiclass": y_val_multiclass,
        "X_test": X_test_scaled, "y_test_multiclass": y_test_multiclass,
        "customers_train": customers_train, "customers_val": customers_val, "customers_test": customers_test,
        "products_train": products_train, "products_val": products_val, "products_test": products_test
    }


# Helper function to extract specific columns for embedding
def extract_column(column_index):
    return lambda x: x[:, column_index]


# Load merged and catalog data
merged_data = pd.read_csv('../processed_data/merged_data.csv')
product_catalog = pd.read_csv('../data/product_catalog.csv')
product_category_map = pd.read_csv('../data/product_category_map.csv')
customer_features = pd.read_csv('../processed_data/customer_features.csv')

# Use a subset of data based on the specified size
size10 = int(size * len(merged_data))
merged_data = merged_data[:size10]
print("INPUT DATA LENGTH:", len(merged_data))

# Encode categorical columns (customer_id and product_id)
customer_encoder = LabelEncoder()
product_encoder = LabelEncoder()

customer_features['customer_id'] = customer_encoder.fit_transform(customer_features['customer_id'])
product_catalog['product_id'] = product_encoder.fit_transform(product_catalog['product_id'])

merged_data['customer_id'] = customer_encoder.transform(merged_data['customer_id'])
merged_data['product_id'] = product_encoder.transform(merged_data['product_id'])

# Save encoders for future use
joblib.dump(customer_encoder, "../encoders/customer_encoder.pkl")
joblib.dump(product_encoder, "../encoders/product_encoder.pkl")
print("Encoders SAVED!")

# Generate sequences for the model
X, y_multiclass, customers, products = create_sequences(
    merged_data, product_catalog, customer_features, n_steps, n_future
)

# Define customer embedding inputs
customer_input = Input(shape=(1,), name='customer_input')  # Single ID for embedding
customer_input_dim = len(np.unique(customers[:, 0]))  # Number of unique customer IDs
customer_output_dim = calculate_output_dim(customer_input_dim)  # Calculate embedding dimension
customer_embedding = Embedding(input_dim=customer_input_dim, output_dim=customer_output_dim)(customer_input)
customer_flat = Flatten()(customer_embedding)

# Normalize customer features and save scaler
scaler = MinMaxScaler()
customers[:, 1:5] = scaler.fit_transform(customers[:, 1:5].astype(float))  # Normalize features
joblib.dump(scaler, "../scalers/customer_features_scaler.pkl")
print("Customer features scaler saved!")

# Define input for customer numerical features
customer_features_input = Input(shape=(4,), name='customer_features_input')  # 4 numerical features
customer_features_dense = Dense(units=32, activation='relu')(customer_features_input)

# Combine embedding and dense layers for customer features
customer_combined = Concatenate()([customer_flat, customer_features_dense])

# Define product embedding inputs
product_input = Input(shape=(7,), name='product_input')  # 7 product-related features
product_embeddings = []

# Create embeddings for each product feature
for i in range(7):
    product_input_dim = len(np.unique(products[:, i]))
    product_output_dim = calculate_output_dim(product_input_dim)
    product_embedding = Embedding(
        input_dim=product_input_dim,
        output_dim=product_output_dim
    )(Lambda(extract_column(i))(product_input))
    product_embeddings.append(Flatten()(product_embedding))
product_combined = Concatenate()(product_embeddings)

# Define time series input for LSTM
time_series_input = Input(shape=(n_steps, 1), name='time_series_input')

# Build LSTM layers
x = LSTM(32, return_sequences=True)(time_series_input)
x = LSTM(16)(x)
x = Dense(16, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(8, activation='relu')(x)
x = Dropout(0.2)(x)

# Combine all inputs (customer, product, time series)
combined = Concatenate()([customer_combined, product_combined, x])

# Define output layer for multiclass classification
multiclass_output = Dense(5, activation='softmax')(combined)

# Create the model
multiclass_model = Model(inputs=[customer_input, customer_features_input, product_input, time_series_input],
                         outputs=multiclass_output)

# Split the data into training, validation, and test sets
splits = split_data(X, y_multiclass, customers, products)

# Display class distribution for training data
unique, counts = np.unique(splits["y_train_multiclass"], return_counts=True)
print("MULTICLASS Y VALUES:", dict(zip(unique, counts)))

# Compile the model with Adam optimizer and sparse categorical crossentropy loss
optimizer = Adam(learning_rate=0.001)
multiclass_model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Compute class weights for imbalanced data
class_weights_multiclass = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(splits["y_train_multiclass"]),
    y=splits["y_train_multiclass"].astype(int)
)
class_weights_dict_multiclass = {i: weight for i, weight in enumerate(class_weights_multiclass)}
print("Class Weights:", class_weights_dict_multiclass)

# Display model summary
multiclass_model.summary()

# Define ModelCheckpoint to save the best model
checkpoint = ModelCheckpoint(
    filepath="../models/best_multiclass_model1.keras",
    monitor='val_loss',  # Monitor validation loss
    save_best_only=True,  # Save only the best model
    save_weights_only=False,  # Save the entire model
    mode='min',  # Minimize validation loss
    verbose=1  # Log model saving process
)

# Train the model
history = multiclass_model.fit(
    [splits["customers_train"][:, 0], splits["customers_train"][:, 1:5], splits["products_train"], splits["X_train"]],
    splits["y_train_multiclass"],
    validation_data=(
        [splits["customers_val"][:, 0], splits["customers_val"][:, 1:5], splits["products_val"], splits["X_val"]],
        splits["y_val_multiclass"]
    ),
    epochs=10,
    batch_size=1024,
    class_weight=class_weights_dict_multiclass,
    callbacks=[checkpoint]  # Use ModelCheckpoint as a callback
)

# Evaluate the model and generate predictions
y_multiclass_pred = multiclass_model.predict(
    [splits["customers_test"][:, 0], splits["customers_test"][:, 1:5], splits["products_test"], splits["X_test"]]
)
y_multiclass_pred_classes = np.argmax(y_multiclass_pred, axis=1)  # Get predicted classes

# Print classification report
print("Classification Report:")
print(classification_report(splits["y_test_multiclass"], y_multiclass_pred_classes))

# Plot confusion matrix
cm_multiclass = confusion_matrix(splits["y_test_multiclass"], y_multiclass_pred_classes)
sns.heatmap(cm_multiclass, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
