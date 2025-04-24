import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array, NotFittedError

# Assuming splines.py is in the same directory
from .splines import (
    calculate_pdp, fit_spline_to_pdp, transform_feature_with_spline,
    calculate_categorical_effects, merge_categories, transform_categorical_feature
)

class XSplinerTransformer(BaseEstimator, TransformerMixin):
    """
A scikit-learn compatible transformer that applies spline transformations 
    based on a black-box model's partial dependence.

    This transformer mimics the core idea of the R xspliner package by:
    1. Calculating Partial Dependence Plots (PDP) for specified numeric features 
       using a provided black-box model.
    2. Fitting Generalized Additive Models (GAMs) with spline terms to these PDPs.
    3. Transforming the numeric features using the fitted GAMs.
    4. Handling categorical features by calculating average model predictions 
       per category and mapping the categories to these values (or merged groups - TBD).

    Parameters
    ----------
    model : object
        The fitted black-box model object. Must have a `predict` or 
        `predict_proba` method.
    
    numeric_features : list of str
        List of names of numeric features to transform using splines.
        
    categorical_features : list of str, optional
        List of names of categorical features to transform. 
        
    xf_params : dict, optional
        Parameters for categorical feature handling. Example:
        {'merge_method': 'none', 'merge_args': {}}. 
        Defaults to {'merge_method': 'none'}. (Merging not yet implemented).

    monotonicity_map : dict, optional
        A dictionary mapping feature names to monotonicity constraints 
        for the spline fitting. Keys are feature names from `numeric_features`, 
        values are 'inc', 'dec', or None. Example: {'age': 'inc', 'bmi': None}.
        Defaults to None (no constraints).

    pdp_params : dict, optional
        Additional keyword arguments passed to `sklearn.inspection.partial_dependence`.
        Example: {'grid_resolution': 50}.
        Defaults to {}.

    gam_params : dict, optional
        Additional keyword arguments passed to the `pygam.LinearGAM` constructor 
        and the `pygam.s` spline term generator. Use a nested dict `{'s_kwargs': {...}}`
        to pass arguments specifically to `s()`. 
        Example: {'n_splines': 10, 'spline_order': 3, 's_kwargs': {'lam': 0.6}}.
        Defaults to {}.
        
    alter_map : dict, optional
        A dictionary mapping feature names to alter behavior ('always', 'never', 
        'auto'). Keys are feature names from `numeric_features`. 
        Defaults to applying splines 'always' for all features.

    compare_stat : str, default='rmse'
        Statistic used for 'auto' comparisons in `alter_map`. Currently 
        supports 'rmse'. 'aic'/'bic' might be added later.
        
    keep_original : bool, default=False
        If True, the original columns specified in `numeric_features` and 
        `categorical_features` are kept in the output DataFrame alongside the
        transformed columns. If False, they are replaced.

    Attributes
    ----------
    fitted_gams_ : dict
        A dictionary storing the fitted pygam GAM objects for each numeric feature.
        Keys are feature names, values are fitted GAM models.
        
    categorical_mappings_ : dict
        Stores the learned mapping for each categorical feature.
        Keys are feature names, values are dictionaries mapping original 
        categories to transformed values.
        
    feature_names_in_ : np.ndarray
        Names of features seen during fit.
        
    n_features_in_ : int
        Number of features seen during fit.

    skipped_features_ : set
        A set of features where spline transformation was skipped.
    """
    def __init__(self, model, numeric_features, categorical_features=None,
                 monotonicity_map=None, pdp_params=None, gam_params=None, 
                 alter_map=None, compare_stat='rmse', 
                 xf_params=None, # Add xf_params
                 keep_original=False):
        self.model = model
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features if categorical_features else []
        self.monotonicity_map = monotonicity_map if monotonicity_map else {}
        self.pdp_params = pdp_params if pdp_params else {}
        self.gam_params = gam_params if gam_params else {}
        self.alter_map = alter_map if alter_map else {}
        self.compare_stat = compare_stat
        self.xf_params = xf_params if xf_params else {'merge_method': 'none'} # Default xf_params
        self.keep_original = keep_original
        # Store target name if possible - needed for categorical effect calculation exclusion
        # This assumes y is passed to fit, or model has a known target attribute.
        # A more robust way might be needed.
        self._target_name = None 

    def fit(self, X, y=None):
        """
        Fit the transformer by calculating PDPs and fitting GAMs.

        Parameters
        ----------
        X : pd.DataFrame
            Training data, where columns are features.
        y : pd.Series or np.ndarray, optional
            Target variable. Not directly used in fitting splines but might be 
            needed by the black-box model's predict function or future categorical methods.
            Defaults to None.

        Returns
        -------
        self
            The fitted transformer instance.
        """
        # Basic input validation
        if not hasattr(self.model, "predict") and not hasattr(self.model, "predict_proba"):
            raise TypeError("Provided model must have a predict or predict_proba method.")
        
        if not isinstance(X, pd.DataFrame):
             # Allow numpy array? For now require DataFrame to easily access columns by name
             raise ValueError("Input X must be a pandas DataFrame.")

        X_ = X.copy()
        
        # Try to infer target name if y is provided
        if isinstance(y, pd.Series):
            self._target_name = y.name
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
             self._target_name = y.columns[0]
             
        # Pass y name to categorical effect calculation if available
        data_for_cat_effects = X_.copy()
        if y is not None and self._target_name:
             # Temporarily join y for effect calculation if needed by method (not current one)
             pass # Current method uses model.predict(X) which shouldn't include target
             
        self.feature_names_in_ = X_.columns.to_numpy()
        self.n_features_in_ = X_.shape[1]
        
        self.fitted_gams_ = {}
        self.skipped_features_ = set() # Store features where spline was skipped (alter='auto')
        self.categorical_mappings_ = {} # Changed from categorical_encoders_

        # Fit splines for numeric features
        for feature in self.numeric_features:
            if feature not in X_.columns:
                raise ValueError(f"Numeric feature '{feature}' not found in input data columns: {list(X_.columns)}")
                
            print(f"Fitting spline for feature: {feature}")
            
            # 1. Calculate PDP
            try:
                pdp_values, avg_responses = calculate_pdp(
                    self.model, X_, feature, **self.pdp_params
                )
            except Exception as e:
                raise RuntimeError(f"Error calculating PDP for feature '{feature}': {e}") from e

            # 2. Fit GAM to PDP (with new options)
            monotonicity = self.monotonicity_map.get(feature, None)
            alter = self.alter_map.get(feature, 'always') # Default to 'always'
            try:
                gam = fit_spline_to_pdp(
                    pdp_values, avg_responses, feature, 
                    monotonic=monotonicity, 
                    alter=alter,
                    compare_stat=self.compare_stat,
                    **self.gam_params
                )
                if gam is not None:
                    self.fitted_gams_[feature] = gam
                else:
                    # Record that this feature's spline was skipped
                    self.skipped_features_.add(feature)
                    print(f"  -> Spline transformation will be skipped for {feature}.")
                    
            except Exception as e:
                raise RuntimeError(f"Error fitting GAM for feature '{feature}': {e}") from e

        # Fit transformations for categorical features
        default_xf_merge_method = self.xf_params.get('merge_method', 'none')
        default_xf_merge_args = self.xf_params.get('merge_args', {})
        
        for feature in self.categorical_features:
            if feature not in X_.columns:
                raise ValueError(f"Categorical feature '{feature}' not found in input data columns: {list(X_.columns)}")
            print(f"Fitting categorical transformation for feature: {feature}")
            
            # 1. Calculate effects per category
            effects = calculate_categorical_effects(self.model, data_for_cat_effects, feature, target=self._target_name)
            if effects.empty:
                print(f"  -> Skipping categorical transformation for {feature} due to issues in effect calculation.")
                continue # Skip to next feature
                
            # 2. Merge categories based on effects (using xf_params)
            merge_method = default_xf_merge_method # Could potentially allow per-feature overrides
            merge_args = default_xf_merge_args
            mapping = merge_categories(effects, method=merge_method, **merge_args)
            
            # Store the learned mapping
            self.categorical_mappings_[feature] = mapping
            
        self._is_fitted = True
        return self

    def transform(self, X):
        """
        Transform the data using the fitted splines and categorical transformations.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        Returns
        -------
        pd.DataFrame
            The transformed data.
        """
        check_is_fitted(self, ['fitted_gams_', 'categorical_mappings_', 'skipped_features_'])
        
        if not isinstance(X, pd.DataFrame):
             raise ValueError("Input X must be a pandas DataFrame.")

        X_ = X.copy()
        
        # Check if columns match training columns
        if not all(col in X_.columns for col in self.feature_names_in_):
            missing_cols = set(self.feature_names_in_) - set(X_.columns)
            # Note: This check might be too strict if transform is intended 
            # to work on subsets of columns, but safer for now.
            # Consider adding a parameter to control this behavior.
            raise ValueError(f"Input columns {list(X_.columns)} do not match columns seen during fit. Missing: {missing_cols}")
        
        transformed_cols = set()
        
        # Apply numeric transformations where GAM was fitted
        for feature, gam in self.fitted_gams_.items():
            print(f"Transforming numeric feature: {feature}")
            try:
                transformed_col = transform_feature_with_spline(gam, X_[feature])
                new_col_name = f"{feature}_xs"
                if self.keep_original:
                    X_[new_col_name] = transformed_col
                    transformed_cols.add(new_col_name)
                else:
                    X_[feature] = transformed_col # Overwrite original
                    transformed_cols.add(feature) 
                    # Optionally rename: X_.rename(columns={feature: new_col_name}, inplace=True)
            except Exception as e:
                raise RuntimeError(f"Error transforming feature '{feature}' with spline: {e}") from e
        
        # Handle numeric features where spline was skipped (alter='auto' said no)
        for feature in self.skipped_features_:
             print(f"Skipping transformation for numeric feature: {feature} (Linear fit was better)")
             # If keep_original is False, the original column is already there.
             # If keep_original is True, the original column is also already there.
             # So, no action needed here unless we want to add a suffix like '_orig'
             # if keep_original is True and we want to differentiate.
             pass

        # Apply categorical transformations
        for feature, mapping in self.categorical_mappings_.items():
             print(f"Transforming categorical feature: {feature}")
             if feature not in X_.columns:
                  print(f"  Warning: Categorical feature '{feature}' to transform not found in input. Skipping.")
                  continue 
             try:
                  transformed_col = transform_categorical_feature(X_[feature], mapping)
                  new_col_name = f"{feature}_xf"
                  if self.keep_original:
                       X_[new_col_name] = transformed_col
                  else:
                       # Explicitly drop original column first, then assign transformed
                       original_dtype = X_[feature].dtype # Keep track of original type for safety/logging if needed
                       X_ = X_.drop(columns=[feature])
                       X_[feature] = transformed_col 
                       # Optionally rename: X_.rename(columns={feature: new_col_name}, inplace=True)
             except Exception as e:
                  raise RuntimeError(f"Error transforming categorical feature '{feature}': {e}") from e

        # Decide which columns to return - current logic keeps all columns
        # (original, overwritten, or new _xs cols if keep_original=True)
        # Might need refinement based on desired output structure
        return X_

    # Define tags directly as a dictionary attribute
    _tags = {
        'requires_y': False, # Set to False as y is optional/used indirectly 
        'X_types': ['2darray', 'dataframe'],
        'allow_nan': False, 
        'stateless': False, 
    } 