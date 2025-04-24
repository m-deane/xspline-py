import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence
from sklearn.linear_model import LinearRegression # For alter='auto' comparison
from sklearn.metrics import mean_squared_error # For comparing fits
# Need to install pygam: pip install pygam
from pygam import LinearGAM, s, te # Import necessary components from pygam
# from pygam import LogisticGAM # For classification?


def calculate_pdp(model, data, feature_name, **pdp_kwargs):
    """
    Calculates the Partial Dependence Plot (PDP) for a given feature.

    Args:
        model: The fitted black-box model object (must have a .predict method).
        data (pd.DataFrame): The training data used for the model.
        feature_name (str): The name of the feature to calculate PDP for.
        **pdp_kwargs: Additional keyword arguments passed to sklearn.inspection.partial_dependence.

    Returns:
        tuple: A tuple containing:
            - pdp_values (np.ndarray): The values of the feature for which PDP was computed.
            - average_responses (np.ndarray): The average predicted response at each feature value.
    """
    # Ensure feature_name is a list for partial_dependence function
    try:
        feature_indices = [data.columns.get_loc(feature_name)]
    except KeyError:
        print(f"Error [calculate_pdp]: Feature '{feature_name}' not found in data columns: {list(data.columns)}")
        return None, None # Cannot proceed
        
    # --- Add Debug Print --- 
    print(f"DEBUG [calculate_pdp]: Columns passed to partial_dependence for feature '{feature_name}': {list(data.columns)}")
    # ----------------------
    
    # Calculate partial dependence
    # Note: Adjust 'method' and other parameters as needed based on xspliner options
    try:
        pdp_result = partial_dependence(
            model, # Should be the pipeline
            data,  # Should be X_train (selected features only)
            features=feature_indices, 
            kind='average',  
            **pdp_kwargs
        )
    except ValueError as ve:
        # Catch the specific error to provide more context
        print(f"ERROR [calculate_pdp]: ValueError during partial_dependence for feature '{feature_name}': {ve}")
        print(f"    Data columns were: {list(data.columns)}")
        # If the error is about feature names, print the model's expected names if possible
        if hasattr(model, 'feature_names_in_'):
             print(f"    Model expected features: {list(model.feature_names_in_)}")
        elif hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'feature_names_in_'):
             print(f"    Final estimator expected features: {list(model.steps[-1][1].feature_names_in_)}")
        raise ve # Re-raise the error after printing info
    except Exception as e:
        print(f"ERROR [calculate_pdp]: Unexpected error during partial_dependence for feature '{feature_name}': {e}")
        raise e
    
    # The result structure depends on the number of features
    # For a single feature:
    average_responses = pdp_result['average'][0]
    pdp_values = pdp_result['grid_values'][0] # Correct key for newer sklearn versions
    
    return pdp_values, average_responses


def _fit_single_gam(X, y, monotonic_constraint, gam_params):
    """Helper function to fit a single GAM with optional monotonicity."""
    constraints = None
    if monotonic_constraint == 'inc':
        constraints = 'monotonic_inc'
    elif monotonic_constraint == 'dec':
        constraints = 'monotonic_dec'
        
    term = s(0, constraints=constraints, **gam_params.get('s_kwargs', {}))
    linear_gam_kwargs = {k: v for k, v in gam_params.items() if k != 's_kwargs'}
    
    # TODO: Choose appropriate GAM type based on model response (e.g., LogisticGAM)
    gam = LinearGAM(term, **linear_gam_kwargs).fit(X, y)
    # Calculate MSE first, then take sqrt for RMSE
    mse = mean_squared_error(y, gam.predict(X)) # Remove squared=False
    rmse = np.sqrt(mse) # Calculate sqrt manually
    return gam, rmse

def fit_spline_to_pdp(pdp_values, average_responses, feature_name, 
                        monotonic=None, alter='always', compare_stat='rmse', **gam_params):
    """
    Fits a GAM (spline) to the calculated PDP data, with enhanced options.

    Args:
        pdp_values (np.ndarray): The feature values from PDP.
        average_responses (np.ndarray): The average responses from PDP.
        feature_name (str): The name of the feature.
        monotonic (str, optional): Monotonicity constraint ('inc', 'dec', 'auto', or None). 
                                   Defaults to None.
        alter (str, optional): When to use spline ('always', 'never', 'auto').
                               'auto' compares spline fit to linear fit.
                               Defaults to 'always'.
        compare_stat (str, optional): Statistic for 'auto' comparisons ('rmse', 'aic', 'bic').
                                      Currently only 'rmse' is implemented simply.
                                      Defaults to 'rmse'.
        **gam_params: Additional keyword arguments passed to pygam's GAM constructor 
                        (e.g., n_splines, spline_order) and s() (via 's_kwargs').

    Returns:
        pygam.GAM or None: The fitted GAM object, or None if alter='never' or 
                           if alter='auto' and linear fit is better.
    """
    if alter == 'never':
        print(f"  Skipping spline for {feature_name} (alter='never')")
        return None
        
    # Reshape data for fitting
    X = pdp_values.reshape(-1, 1)
    y = average_responses

    best_gam = None
    best_gam_rmse = np.inf

    # --- Handle Monotonicity ---
    if monotonic == 'auto':
        print(f"  Fitting with monotonic='auto' for {feature_name}...")
        gam_inc, rmse_inc = _fit_single_gam(X, y, 'inc', gam_params)
        gam_dec, rmse_dec = _fit_single_gam(X, y, 'dec', gam_params)
        print(f"    RMSE (increasing): {rmse_inc:.4f}, RMSE (decreasing): {rmse_dec:.4f}")
        if rmse_inc <= rmse_dec:
            best_gam = gam_inc
            best_gam_rmse = rmse_inc
            print(f"    Selected increasing monotonic spline.")
        else:
            best_gam = gam_dec
            best_gam_rmse = rmse_dec
            print(f"    Selected decreasing monotonic spline.")
            
    elif monotonic in ['inc', 'dec', 'up', 'down']:
        mono_constraint = 'inc' if monotonic in ['inc', 'up'] else 'dec'
        print(f"  Fitting with monotonic='{mono_constraint}' for {feature_name}...")
        best_gam, best_gam_rmse = _fit_single_gam(X, y, mono_constraint, gam_params)
        print(f"    RMSE: {best_gam_rmse:.4f}")
    else: # monotonic is None or invalid
        print(f"  Fitting unconstrained spline for {feature_name}...")
        best_gam, best_gam_rmse = _fit_single_gam(X, y, None, gam_params)
        print(f"    RMSE: {best_gam_rmse:.4f}")

    # --- Handle Alter ('auto' vs linear) ---
    if alter == 'auto':
        print(f"  Comparing spline (RMSE={best_gam_rmse:.4f}) vs linear fit for {feature_name}...")
        # Fit linear model to PDP data
        lin_reg = LinearRegression()
        lin_reg.fit(X, y)
        # Calculate MSE first, then take sqrt for RMSE
        lin_mse = mean_squared_error(y, lin_reg.predict(X)) # Remove squared=False
        lin_rmse = np.sqrt(lin_mse) # Calculate sqrt manually
        print(f"    Linear fit RMSE: {lin_rmse:.4f}")
        
        # Compare based on the chosen statistic (currently only RMSE)
        # TODO: Implement AIC/BIC comparison if needed (requires calculating degrees of freedom)
        if compare_stat == 'rmse':
            if lin_rmse <= best_gam_rmse: # If linear is better or equal
                print(f"    Linear fit is better or equal (RMSE). Using bare component for {feature_name}.")
                return None # Indicate not to use spline
            else:
                print(f"    Spline fit is better (RMSE). Using spline transformation.")
                return best_gam
        # elif compare_stat == 'aic': # Placeholder
        #     # Calculate AIC for both models (requires log-likelihood, N, K)
        #     # gam_aic = best_gam.statistics_['AIC'] # pygam might provide this
        #     # lin_aic = ... 
        #     pass 
        else:
             print(f"  Warning: compare_stat '{compare_stat}' not implemented for alter='auto'. Defaulting to using spline.")
             return best_gam
             
    # If alter is 'always' (or 'auto' decided spline is better)
    return best_gam


def transform_feature_with_spline(gam_model, data_column):
    """
    Transforms a data column using the fitted GAM spline.

    Args:
        gam_model (pygam.GAM): The fitted GAM object.
        data_column (pd.Series or np.ndarray): The original feature column to transform.

    Returns:
        np.ndarray: The transformed feature values (spline predictions).
    """
    X_transform = data_column.values.reshape(-1, 1)
    # Use predict to get the spline-transformed value
    transformed_values = gam_model.predict(X_transform)
    return transformed_values

# --- Categorical Feature Handling (xf) --- 

def calculate_categorical_effects(model, data, feature_name, target=None):
    """
    Calculates the average model prediction for each category level.
    (Simplified version of ICE-based effects in R xspliner)

    Args:
        model: Fitted black-box model.
        data (pd.DataFrame): Training data.
        feature_name (str): Name of the categorical feature.
        target (str, optional): Name of target variable (not currently used, but maybe for future methods).

    Returns:
        pd.Series: A series mapping category levels to their average prediction.
    """
    print(f"  Calculating effects for categorical feature: {feature_name}")
    # Group by category and calculate mean prediction
    # Ensure predict method exists and handle potential errors
    if hasattr(model, 'predict'):
        # Need a copy of data to add predictions without modifying original
        data_copy = data.copy()
        try:
            # Use try-except for predict issues
            data_copy['_prediction'] = model.predict(data[[c for c in data.columns if c != target]])
            # Calculate mean prediction per category
            category_effects = data_copy.groupby(feature_name)['_prediction'].mean()
            return category_effects
        except Exception as e:
            print(f"    Warning: Could not get predictions for {feature_name}. Error: {e}. Skipping effects calculation.")
            return pd.Series(dtype=float) # Return empty series
    else:
        print(f"    Warning: Model lacks predict method. Cannot calculate effects for {feature_name}.")
        return pd.Series(dtype=float) # Return empty series
        
def merge_categories(effects, method='none', **kwargs):
    """
    Merges categories based on their effects.
    
    Args:
        effects (pd.Series): Series mapping category levels to effects (e.g., mean predictions).
        method (str): Merging method ('none', 'quantile').
        **kwargs: Additional arguments for the merging method. 
                  For 'quantile', expects 'n_bins' (int).

    Returns:
        dict: A dictionary mapping original category levels to merged group identifiers/values.
    """
    print(f"  Merging categories (method='{method}')...")
    
    if method == 'quantile':
        n_bins = kwargs.get('n_bins', 5) # Default to 5 bins if not provided
        if not isinstance(n_bins, int) or n_bins <= 0:
            print(f"    Warning: Invalid n_bins ({n_bins}) for quantile merging. Defaulting to 5.")
            n_bins = 5
            
        if effects.empty or effects.nunique() <= 1:
            print("    Warning: Cannot perform quantile merging with empty or single-valued effects. Returning identity mapping.")
            return effects.to_dict()
            
        try:
            # Use pd.qcut to bin the effects into quantiles
            # labels=False returns integer indicators (0 to n_bins-1)
            # duplicates='drop' handles cases where quantile edges are not unique
            bins = pd.qcut(effects, q=n_bins, labels=False, duplicates='drop')
            mapping = bins.to_dict()
            print(f"    Merged {len(effects)} categories into {bins.nunique()} quantile bins.")
            return mapping
        except ValueError as e:
            # Handle cases where qcut fails (e.g., too few unique values for n_bins)
            print(f"    Warning: Quantile merging failed ({e}). Returning identity mapping.")
            return effects.to_dict()
        except Exception as e:
            print(f"    Error during quantile merging: {e}. Returning identity mapping.")
            return effects.to_dict()
            
    elif method == 'none':
        # Return identity mapping (original effects)
        print(f"    No merging applied (method='none').")
        return effects.to_dict()
    else:
        print(f"    Warning: Unknown merge method '{method}'. Returning identity mapping.")
        return effects.to_dict()

def transform_categorical_feature(data_column, mapping):
    """
    Transforms a categorical column using a learned mapping.

    Args:
        data_column (pd.Series): The original categorical feature column.
        mapping (dict): Dictionary mapping original categories to transformed values.

    Returns:
        pd.Series: The transformed feature column.
    """
    print(f"  Applying categorical mapping for: {data_column.name}")
    # Map values, handle unseen categories (e.g., fill with mean/median of mapped values or a default)
    transformed = data_column.map(mapping)
    # Simple handling for unseen values: fill with the mean of the mapped values
    if transformed.isnull().any():
        fill_value = np.nanmean(list(mapping.values())) if mapping else 0
        print(f"    Found unseen categories. Filling with: {fill_value:.4f}")
        transformed = transformed.fillna(fill_value)
    return transformed

# TODO: Implement wrapper functions mimicking the R package's xs() and xf() within formulas,
#       or integrate this logic into the scikit-learn transformer. 