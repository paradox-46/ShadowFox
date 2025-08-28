Car Price Prediction Project

This project implements a machine learning solution to predict car prices based on various features such as make, model, year, mileage, and other relevant attributes.

Project Overview
The car price prediction system uses ensemble learning methods (Random Forest and Gradient Boosting) to accurately estimate vehicle prices. The solution includes comprehensive data preprocessing, exploratory data analysis, model training, evaluation, and deployment-ready components.

Features
Data Preprocessing: Handles missing values, encodes categorical variables, and scales numerical features

Exploratory Data Analysis: Generates visualizations including correlation heatmaps, distribution plots, and feature relationships

Model Training: Implements both Random Forest and Gradient Boosting regressors

Model Evaluation: Comprehensive evaluation using multiple metrics (MAE, MSE, RMSE, R²) with cross-validation

Feature Importance Analysis: Identifies the most influential factors in car pricing

Prediction Capability: Ready-to-use model for making predictions on new data

Requirements
Python 3.7+

pandas

numpy

matplotlib

seaborn

scikit-learn

joblib

Install dependencies with:

bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
Usage
Place your car dataset named car_data.csv in the project directory

Run the script:

bash
python car_price_prediction.py
The script will:

Load and preprocess the data

Perform exploratory data analysis

Train and evaluate machine learning models

Generate visualizations of results

Save the best performing model for future use

Expected Dataset Format
The script expects a CSV file with columns that may include:

Car_Name: Name of the car

Year: Manufacturing year

Present_Price: Current showroom price (in lakhs)

Kms_Driven: Distance driven in kilometers

Fuel_Type: Type of fuel (Petrol/Diesel/CNG)

Seller_Type: Type of seller (Dealer/Individual)

Transmission: Transmission type (Manual/Automatic)

Owner: Number of previous owners

Selling_Price: Target variable - selling price (in lakhs)

Output Files
eda_plots.png: Exploratory data analysis visualizations

feature_importance.png: Feature importance chart

prediction_vs_actual.png: Model performance visualization

residual_analysis.png: Residual analysis plots

car_price_rf_model.pkl or car_price_gb_model.pkl: Trained model

scaler.pkl: Feature scaler for new data

feature_names.pkl: List of feature names

target_name.pkl: Target variable name

Model Performance
The script evaluates models using:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R-squared (R²) score

Cross-validated R² score

Making New Predictions
After training, the model can be used to predict prices for new cars:

python
import joblib

# Load the saved model
model = joblib.load('car_price_rf_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

# Prepare new data (ensure it has the same features as training)
new_car_data = pd.DataFrame([{
    'Year': 2017,
    'Present_Price': 7.5,
    'Kms_Driven': 35000,
    # ... other features
}])

# Scale the features
new_car_scaled = scaler.transform(new_car_data)

# Make prediction
predicted_price = model.predict(new_car_scaled)[0]
print(f"Predicted Selling Price: ₹{predicted_price:.2f} Lakhs")
Customization
You can modify the script to:

Adjust model hyperparameters

Add new features or preprocessing steps

Incorporate different machine learning algorithms

Change visualization styles

Modify the train-test split ratio
