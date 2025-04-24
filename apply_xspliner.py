import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # Example black-box model
from sklearn.linear_model import LinearRegression # Example final model
from sklearn.metrics import mean_squared_error

# Assuming xspliner_py package is installable or in the PYTHONPATH
from xspliner_py.transformer import XSplinerTransformer
# Import PDP calculation function directly for plotting
from xspliner_py.splines import calculate_pdp 

# --- Configuration ---
DATA_FILE = '__data/preem.csv'
TARGET_VARIABLE = 'target'
TEST_SIZE = 0.3
RANDOM_STATE = 42
PLOT_DIR = 'plots' # Directory to save plots

# Create plot directory if it doesn't exist
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
    print(f"Created directory: {PLOT_DIR}")

# --- Load Data ---
try:
    data = pd.read_csv(DATA_FILE)
    print(f"Successfully loaded data from {DATA_FILE}")
    print(f"Data shape: {data.shape}")
    print("Data head:\n", data.head())
except FileNotFoundError:
    print(f"Error: Could not find the data file at {DATA_FILE}.")
    print("Please ensure the file exists in the current directory.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit()

# --- Data Preparation ---
if TARGET_VARIABLE not in data.columns:
    print(f"Error: Target variable '{TARGET_VARIABLE}' not found in the data.")
    exit()

y = data[TARGET_VARIABLE]
X = data.drop(columns=[TARGET_VARIABLE])

# --- Identify Feature Types ---
# Convert date to string if not already, just in case
X['date'] = X['date'].astype(str)

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

print(f"\nIdentified Numeric Features: {numeric_features}")
print(f"Identified Categorical Features: {categorical_features}")

# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"\nSplit data into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples)")

# --- Train Black-Box Model ---
# Need to handle the categorical 'date' column before training RF
# Simple approach: OneHotEncode the date column for the RF model
# More complex feature engineering could be done

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Create preprocessor for the black-box model
# OneHotEncode 'date', pass through numeric features
# handle_unknown='ignore' deals with dates potentially only in test set
preprocessor_rf = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)],
    remainder='passthrough' # Keep numeric features as they are
)

print("\nTraining black-box model (RandomForestRegressor with OHE for date)...")
# Create a pipeline with preprocessing and the model
black_box_model_pipeline = Pipeline(steps=[('preprocessor', preprocessor_rf),
                                           ('regressor', RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE))])

black_box_model_pipeline.fit(X_train, y_train)
print("Black-box model trained.")

# --- Instantiate and Fit XSplinerTransformer ---
# Use alter='auto' and monotonic='auto' for numeric features
alter_map = {feature: 'auto' for feature in numeric_features}
monotonicity_map = {feature: 'auto' for feature in numeric_features} 

# Add back definitions for pdp_params and gam_params
pdp_params = {'grid_resolution': 20} 
gam_params = {'n_splines': 8} 

# Configure categorical handling (quantile binning)
xf_params = {
    'merge_method': 'quantile', 
    'merge_args': {'n_bins': 5} # Example: 5 bins
}

print("\nInstantiating XSplinerTransformer...")
xspliner = XSplinerTransformer(
    model=black_box_model_pipeline, # Pass the whole pipeline
    numeric_features=numeric_features,
    categorical_features=categorical_features, # Now includes 'date'
    monotonicity_map=monotonicity_map,
    alter_map=alter_map,
    compare_stat='rmse',
    pdp_params=pdp_params, # Now defined
    gam_params=gam_params, # Now defined
    xf_params=xf_params, 
    keep_original=False # Replace original features with transformed ones
)

print("Fitting XSplinerTransformer...")
# Pass the original X_train (without OHE) to the transformer
# The transformer's PDP/effect calculations need the original features
try:
    xspliner.fit(X_train, y_train)
    print("XSplinerTransformer fitted successfully.")
except NotImplementedError as e:
    print(f"Fitting failed: {e}")
    print("Categorical feature handling might not be fully implemented yet.")
    # Optionally continue without categorical features if desired
    # print("Retrying fit without categorical features...")
    # xspliner.categorical_features = []
    # xspliner.fit(X_train, y_train)
    # print("XSplinerTransformer fitted without categorical features.")
except Exception as e:
    print(f"An error occurred during XSplinerTransformer fitting: {e}")
    exit()

# --- Plot PDP and Fitted Splines ---
print(f"\nGenerating and saving plots to {PLOT_DIR}/...")
if hasattr(xspliner, 'fitted_gams_') and xspliner.fitted_gams_:
    for feature, gam in xspliner.fitted_gams_.items():
        print(f"Plotting for feature: {feature}")
        try:
            # Recalculate PDP for plotting (or retrieve if stored)
            # Using X_train for PDP calculation context
            pdp_values, avg_responses = calculate_pdp(
                xspliner.model, X_train, feature, **xspliner.pdp_params
            )
            
            # Generate points for GAM curve
            gam_x = np.linspace(pdp_values.min(), pdp_values.max(), 200).reshape(-1, 1)
            gam_y = gam.predict(gam_x)
            
            plt.figure(figsize=(10, 6))
            # Plot PDP points
            plt.scatter(pdp_values, avg_responses, label='PDP Average Response', alpha=0.7, s=50)
            # Plot fitted GAM spline
            plt.plot(gam_x, gam_y, label='Fitted GAM Spline', color='red', linewidth=2)
            
            plt.title(f'PDP and Fitted Spline for: {feature}')
            plt.xlabel(feature)
            plt.ylabel('Partial Dependence (Target Scale)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # Sanitize filename
            safe_feature_name = "".join(c if c.isalnum() else '_' for c in feature)
            plot_filename = os.path.join(PLOT_DIR, f'pdp_gam_{safe_feature_name}.png')
            plt.savefig(plot_filename)
            plt.close() # Close the plot to free memory
            print(f"  Saved plot: {plot_filename}")

        except Exception as e:
            print(f"  Error plotting feature '{feature}': {e}")
else:
    print("No fitted GAMs found in the transformer to plot.")

# --- Transform Data ---
print("\nTransforming training and testing data...")
try:
    X_train_transformed = xspliner.transform(X_train)
    X_test_transformed = xspliner.transform(X_test)
    print("Data transformed successfully.")
    print("\nTransformed training data head:\n", X_train_transformed.head())
except Exception as e:
    print(f"An error occurred during data transformation: {e}")
    exit()

# --- Train and Evaluate Final Model (e.g., Linear Regression) ---
print("\nTraining final model (LinearRegression) on transformed data...")
final_model = LinearRegression()

# Ensure columns used for fitting match columns available after transformation
# X_train_transformed now contains transformed numeric + transformed categorical ('date')
final_model.fit(X_train_transformed, y_train)
print("Final model trained.")

print("\nEvaluating final model...")
y_pred = final_model.predict(X_test_transformed)
mse = mean_squared_error(y_test, y_pred)
score = final_model.score(X_test_transformed, y_test) # R^2 score for regression

print(f"Final Model (LinearRegression on transformed data) Performance:")
print(f"  Mean Squared Error (MSE): {mse:.4f}")
print(f"  R-squared Score: {score:.4f}")

print("\nScript finished.") 