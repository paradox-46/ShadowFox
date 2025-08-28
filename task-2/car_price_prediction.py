# car_price_prediction.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("Loading and preprocessing car data...")

# Load the dataset
try:
    df = pd.read_csv('car_data.csv')
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'car_data.csv' not found. Please download the dataset and place it in the same directory.")
    exit()

# Display basic information about the dataset
print("\nDataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check for duplicate rows
print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")

# Data Preprocessing
print("\nPreprocessing data...")

# Handle missing values
if df.isnull().sum().sum() > 0:
    print("Handling missing values...")
    # Impute numerical columns with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='median')
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

# Handle categorical variables
print("Encoding categorical variables...")
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        # Handle missing values in categorical columns first
        df[col] = df[col].fillna('Unknown')
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded {col}: {dict(enumerate(le.classes_))}")

# Define features (X) and target (y)
# Let's first identify which column is likely the target
print("\nIdentifying target variable...")
# Common target names for car price prediction
possible_targets = ['Selling_Price', 'SellingPrice', 'Price', 'Selling Cost', 'Selling_Amount']

target_col = None
for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    # If none of the common names found, try to guess
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Assume the last numeric column might be the target
    target_col = numeric_cols[-1]
    print(f"Target column not explicitly found. Using '{target_col}' as target.")

print(f"Using '{target_col}' as target variable.")

# Remove non-predictive columns
columns_to_drop = ['Car_Name', 'Unnamed: 0', 'index']  # Common non-predictive columns
X = df.drop([col for col in columns_to_drop if col in df.columns] + [target_col], axis=1, errors='ignore')
y = df[target_col]

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature names: {X.columns.tolist()}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Scale the features (only numerical columns)
numerical_features = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

if len(numerical_features) > 0:
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

# Exploratory Data Analysis (EDA)
print("\nPerforming exploratory data analysis...")
plt.figure(figsize=(15, 10))

# Correlation heatmap
plt.subplot(2, 2, 1)
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', center=0)
plt.title('Feature Correlation Matrix')

# Target distribution
plt.subplot(2, 2, 2)
sns.histplot(y, kde=True)
plt.title('Distribution of Selling Prices')
plt.xlabel('Selling Price')

# Boxplot of target variable
plt.subplot(2, 2, 3)
sns.boxplot(y=y)
plt.title('Boxplot of Selling Prices')
plt.xlabel('Selling Price')

# Check relationship with a likely important feature
if 'Present_Price' in df.columns:
    plt.subplot(2, 2, 4)
    plt.scatter(df['Present_Price'], y, alpha=0.6)
    plt.xlabel('Present Price')
    plt.ylabel('Selling Price')
    plt.title('Present Price vs Selling Price')
else:
    # Use the first numerical feature instead
    first_num_col = df.select_dtypes(include=[np.number]).columns[0]
    if first_num_col != target_col:
        plt.subplot(2, 2, 4)
        plt.scatter(df[first_num_col], y, alpha=0.6)
        plt.xlabel(first_num_col)
        plt.ylabel('Selling Price')
        plt.title(f'{first_num_col} vs Selling Price')

plt.tight_layout()
plt.savefig('eda_plots.png')
plt.show()

# Model Training
print("\nTraining models...")

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train_scaled, y_train)

# Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
gb_model.fit(X_train_scaled, y_train)

# Model Evaluation
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    print(f"\n{model_name} Performance:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    print(f"Cross-validated R-squared: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
    
    return y_pred, mae, rmse, r2, cv_scores

# Evaluate both models
print("\n" + "="*50)
rf_pred, rf_mae, rf_rmse, rf_r2, rf_cv = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
print("\n" + "="*50)
gb_pred, gb_mae, gb_rmse, gb_r2, gb_cv = evaluate_model(gb_model, X_test_scaled, y_test, "Gradient Boosting")

# Feature Importance Analysis
print("\nAnalyzing feature importance...")

# Get feature names that match the number of importance values
feature_names = X.columns.tolist()
n_features = len(feature_names)
n_importance = len(rf_model.feature_importances_)

print(f"Number of features: {n_features}")
print(f"Number of importance values: {n_importance}")

# Handle the case where they might not match
if n_features != n_importance:
    print("Warning: Number of features doesn't match number of importance values!")
    print("Using the first features that match the importance values")
    if n_features > n_importance:
        feature_names = feature_names[:n_importance]
    else:
        # This shouldn't happen, but just in case
        feature_names = feature_names + [f'feature_{i}' for i in range(n_features, n_importance)]

# Now create the Series
feature_importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(12, 8))
feature_importances.plot(kind='bar')
plt.title('Feature Importances for Predicting Car Price')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

print("\nTop 5 most important features:")
for feature, importance in feature_importances.head().items():
    print(f"{feature}: {importance:.4f}")

# Visualization of predictions vs actual values
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, rf_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Random Forest: Actual vs Predicted Prices')
plt.text(0.05, 0.95, f'R² = {rf_r2:.3f}', transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8))

plt.subplot(1, 2, 2)
plt.scatter(y_test, gb_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Gradient Boosting: Actual vs Predicted Prices')
plt.text(0.05, 0.95, f'R² = {gb_r2:.3f}', transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('prediction_vs_actual.png')
plt.show()

# Residual analysis
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
residuals_rf = y_test - rf_pred
plt.scatter(rf_pred, residuals_rf, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Random Forest: Residual Plot')

plt.subplot(1, 2, 2)
residuals_gb = y_test - gb_pred
plt.scatter(gb_pred, residuals_gb, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Gradient Boosting: Residual Plot')

plt.tight_layout()
plt.savefig('residual_analysis.png')
plt.show()

# Making a prediction on new data
print("\nMaking prediction on sample new car data...")

# First, let's see what features we have
print(f"Available features: {X.columns.tolist()}")

# Create sample data based on available features
sample_data = {}
for feature in X.columns:
    if feature in df.columns:
        # Use median value for numerical features
        if df[feature].dtype in [np.int64, np.float64]:
            sample_data[feature] = df[feature].median()
        # Use mode for categorical features
        else:
            sample_data[feature] = df[feature].mode()[0]

# Update with some realistic values if we know what they represent
if 'Year' in sample_data:
    sample_data['Year'] = 2017
if 'Present_Price' in sample_data:
    sample_data['Present_Price'] = 7.5
if 'Kms_Driven' in sample_data:
    sample_data['Kms_Driven'] = 35000

print(f"\nSample data: {sample_data}")

# Create a DataFrame from the sample data
sample_df = pd.DataFrame([sample_data])

# Ensure we have the same columns as training data
sample_df = sample_df[X.columns]

# Scale the numerical features
sample_scaled = sample_df.copy()
if len(numerical_features) > 0:
    sample_scaled[numerical_features] = scaler.transform(sample_df[numerical_features])

# Predict with both models
rf_predicted_price = rf_model.predict(sample_scaled)[0]
gb_predicted_price = gb_model.predict(sample_scaled)[0]

print(f"\nSample Car Prediction:")
print(f"Random Forest Predicted Selling Price: ₹{rf_predicted_price:.2f} Lakhs")
print(f"Gradient Boosting Predicted Selling Price: ₹{gb_predicted_price:.2f} Lakhs")

# Save the best model
import joblib

if rf_r2 > gb_r2:
    joblib.dump(rf_model, 'car_price_rf_model.pkl')
    print("\nRandom Forest model saved as 'car_price_rf_model.pkl' (better performance)")
else:
    joblib.dump(gb_model, 'car_price_gb_model.pkl')
    print("\nGradient Boosting model saved as 'car_price_gb_model.pkl' (better performance)")

# Save the scaler and feature names
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')
joblib.dump(target_col, 'target_name.pkl')

print("\nData preprocessing objects saved.")
print("Car price prediction completed successfully!")