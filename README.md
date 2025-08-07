# 🏠 California Housing Price Prediction

This project builds and deploys a machine learning model to predict housing prices in California using the classic `housing.csv` dataset from the book *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*.

## 📁 Project Structure

- `housing.csv`: Original dataset with California housing data.
- `main_old.py`: Exploratory and model comparison script using Linear, Decision Tree, and Random Forest regression.
- `main.py`: Production-ready pipeline for training or inference with Random Forest Regressor.
- `input.csv`: Saved test dataset used during the inference phase.
- `output.csv`: Final predictions appended to the test dataset.
- `model.pkl`: Trained model saved for reuse.
- `pipeline.pkl`: Preprocessing pipeline used for consistent transformation.

## 🔧 Workflow

### 1. Training Phase (`main.py`)
- Loads the dataset.
- Performs **stratified sampling** based on income categories.
- Builds a pipeline for:
  - Numerical attributes (imputation + standardization)
  - Categorical attributes (`OneHotEncoder`)
- Trains a `RandomForestRegressor`.
- Saves both the model and the pipeline.

### 2. Inference Phase (`main.py`)
- Loads the saved model and pipeline.
- Applies preprocessing to the test set (`input.csv`).
- Generates predictions for `median_house_value`.
- Saves results to `output.csv`.

### 3. Model Comparison (`main_old.py`)
- Compares performance of:
  - Linear Regression
  - Decision Tree Regression
  - Random Forest Regression
- Uses 10-fold cross-validation with **Root Mean Squared Error (RMSE)** as the evaluation metric.
- Outputs basic descriptive statistics of errors for each model.

## 🚀 How to Run

**Train & Save Model (if not already saved):**
```bash
python main.py
```

**Run Inference (once model is saved):**
```bash
python main.py
```

> The script will detect whether the model and pipeline exist, and switch between training or inference accordingly.

## 📦 Requirements

Make sure to install the following Python libraries:
```bash
pip install pandas numpy scikit-learn joblib
```

## 📊 Dataset Features

- `longitude`, `latitude`: Geographical location
- `housing_median_age`: Age of the houses
- `total_rooms`, `total_bedrooms`, `population`, `households`
- `median_income`: Median income in the area
- `ocean_proximity`: Categorical feature indicating proximity to the ocean
- `median_house_value`: **Target variable**

## 📈 Output

The final output is a CSV file (`output.csv`) containing the test data along with a new column `median_house_value` containing the predicted prices.
