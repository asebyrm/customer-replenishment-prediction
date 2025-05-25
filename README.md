# Customer Replenishment Prediction

This project implements a Long Short-Term Memory (LSTM) model for multiclass prediction of customer behavior based on time-series data, product features, and customer features. The goal is to predict the likelihood of a purchase in the upcoming weeks.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Data Preparation](#data-preparation)
5. [Model Training](#model-training)
6. [Prediction](#prediction)
7. [Results](#results)
8. [How to Run](#how-to-run)
9. [File Structure](#file-structure)
10. [Acknowledgements](#acknowledgements)

---

## Overview

This project uses an LSTM neural network for multiclass classification. The model predicts the purchase behavior of customers based on time-series transactional data, combined with customer and product metadata.

The key objectives are:
- Process historical transaction data to generate time-series inputs.
- Integrate customer and product features.
- Train a multiclass classification model.
- Make predictions for upcoming weeks.

---

## Features

- **Data Preprocessing**:
  - Encodes customer and product IDs.
  - Scales numerical features.
  - Extracts relevant sequences for LSTM inputs.

- **Model Architecture**:
  - LSTM layers for sequential data.
  - Embedding layers for customer and product features.
  - Dense layers for feature integration.
  - Output layer with softmax activation for multiclass prediction.

- **Prediction and Evaluation**:
  - Generates predictions for test data.
  - Provides classification reports and confusion matrices.
  - Saves results to CSV for further analysis.

---

## Requirements

To run this project, you need the following:

- Python 3.10+
- TensorFlow 2.6+
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- joblib

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## Data Preparation

The project requires the following input datasets:

1. **Transaction Data**:
   - `data/transactions.csv`: Contains customer-product weekly purchase quantities.

2. **Product Catalog**:
   - `data/product_catalog.csv`: Contains product-specific features such as `category_id`.

3. **Customer Features**:
   - `processed_data/customer_features.csv`: Contains customer-specific numerical and categorical features.

4. **Test Data**:
   - `data/test.csv`: Contains customer-product pairs for prediction.

### Preprocessing Steps

- Encode `customer_id` and `product_id`.
- Normalize numerical features.
- Prepare sequences for LSTM inputs.

---

## Model Training

The model architecture includes:

- **Customer Embeddings**:
  - Input: Encoded customer IDs and numerical features.
  - Output: Dense representation of customer data.

- **Product Embeddings**:
  - Input: Encoded product IDs and product features.
  - Output: Dense representation of product data.

- **Time-Series Data**:
  - Input: Weekly purchase quantities.
  - Output: Sequential representation using LSTM layers.

### Training Parameters

- **Loss Function**: Sparse Categorical Cross-entropy
- **Optimizer**: Adam (learning rate = 0.001)
- **Metrics**: Accuracy
- **Batch Size**: 1024
- **Epochs**: 10
- **Class Weights**: Balanced based on target distribution

---

## Prediction

The trained model generates predictions for test data. Predictions include the likelihood of a purchase occurring in the next 1–4 weeks. 

Predictions are saved to a CSV file:
```
predictions/predictions.csv
```

Columns:
- `customer_id`: Original customer ID.
- `product_id`: Original product ID.
- `prediction`: Predicted class (0-4).

---

## Results

- **Classification Report**:
  - Precision, Recall, F1-Score for each class.

- **Confusion Matrix**:
  - Visualized using Seaborn heatmap.

- **Example Predictions**:
  ```plaintext
  customer_id, product_id, prediction
  12345, 67890, 2
  54321, 98765, 0
  ```

---

## How to Run


1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare the input data in the required format.

3. Train the model:
   ```bash
   python LstmCodes/forecasting_multiVII.py
   ```

4. Generate predictions:
   ```bash
   python notebooks/predicter.py
   ```

---

## File Structure

```
LSTMV502/
├── .venv/                    # Virtual environment files
├── data/                     # Raw input data
│   ├── product_catalog.csv
│   ├── product_category_map.csv
│   ├── test.csv
│   ├── transactions.csv
├── encoders/                 # Encoded customer and product IDs
│   ├── customer_encoder.pkl
│   ├── product_encoder.pkl
├── LstmCodes/                # Python scripts for preprocessing and modeling
│   ├── customerfeatures.py
│   ├── forecasting_multiVII.py
│   ├── preprocess.py
│   ├── productfeatures.py
├── models/                   # Trained models
│   ├── best_multiclass_model1.keras
│   ├── multiclass_model.keras
├── notebooks/                # Jupyter notebooks for analysis
│   ├── corranalize.ipynb
│   ├── customer_features_analyze.ipynb
│   ├── data_analyze.ipynb
│   ├── hierarchicalanalyze.ipynb
│   ├── predicter.ipynb
│   ├── similarity.ipynb
├── predictions/              # Prediction outputs
│   ├── predictions.csv
│   ├── predictions_updated4.csv
├── processed_data/           # Processed datasets
│   ├── customer_features.csv
│   ├── final_product_catalog.csv
│   ├── merged_data.csv
│   ├── predictions_updated.csv
│   ├── products_and_categories.csv
├── scalers/                  # Scalers used for normalization
│   ├── customer_features_scaler.pkl
├── README.md                 # Project documentation
├── requirements.txt          # Required Python packages
```

---

## Acknowledgements

- TensorFlow and Keras documentation.
- scikit-learn for data preprocessing.
- Matplotlib and Seaborn for visualizations.

---

If you encounter issues or have questions, feel free to open an issue or reach out via email. (ahmetselim.asb@gmail.com)

