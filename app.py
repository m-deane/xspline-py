# app.py
import base64
import io
import os
import traceback

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, dash_table
from dash.dash_table.Format import Format, Scheme

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import from your xspliner_py library
# Ensure xspliner_py is in the Python path or installed
try:
    from xspliner_py.transformer import XSplinerTransformer
    from xspliner_py.splines import calculate_pdp
except ImportError:
    print("Error: Could not import from xspliner_py.")
    print("Ensure the library is installed or accessible in your PYTHONPATH.")
    # Add placeholder classes/functions if needed for app layout testing
    class XSplinerTransformer: pass # Placeholder
    def calculate_pdp(*args, **kwargs): return None, None # Placeholder

# --- Helper function for SMAPE ---
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """Calculate SMAPE."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Handle division by zero: replace NaN/inf with 0 or a large number?
    # For simplicity, replacing with 0 where denominator is close to zero.
    ratio = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator!=0)
    return np.mean(ratio) * 100 # Return as percentage

# --- Configuration & Initialization ---
# Use the FLATLY theme from Dash Bootstrap Components
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
server = app.server

# --- Reusable Components ---
def create_parameter_input(label, input_id, tooltip_text, default_value, input_type="number", **kwargs):
    """Helper to create labeled input with tooltip."""
    return dbc.Row(
        [
            dbc.Label(label, html_for=input_id, width=6),
            dbc.Col(
                dbc.Input(id=input_id, type=input_type, value=default_value, **kwargs),
                width=5,
            ),
            dbc.Col(
                html.I(className="bi bi-info-circle-fill", id=f"tooltip-{input_id}"),
                width=1,
            ),
            dbc.Tooltip(tooltip_text, target=f"tooltip-{input_id}"),
        ],
        className="mb-2 align-items-center",
    )

# Define content for the Parameter Reference tab
reference_content = dcc.Markdown("""
### Parameter Reference

This section explains the parameters you can configure in the sidebar.

---        

**1. Black-Box Model Parameters**

*   **Num. Estimators (`input-n-estimators`):** 
    *   Controls the number of trees in the Random Forest or Gradient Boosting model.
    *   *Usage:* More estimators can lead to better performance but increase training time. 

---

**2. XSpliner Global Parameters**

*   **PDP Grid Resolution (`input-grid-res`):** 
    *   The number of grid points used to calculate the Partial Dependence Plot (PDP) for numeric features. The spline is then fitted to these points.
    *   *Usage:* Higher values provide a finer-grained PDP but take longer to compute. 
*   **Num. Splines (GAM) (`input-n-splines`):** 
    *   The number of basis functions used to represent the spline curve by the underlying Generalized Additive Model (GAM).
    *   *Usage:* Controls the flexibility of the fitted spline. Too few may underfit the PDP, too many may overfit or wiggle excessively.
*   **Smoothing (Lambda) (`input-lam`):** 
    *   The penalty strength applied to the GAM spline coefficients to control smoothness (wiggliness).
    *   *Usage:* Higher values lead to smoother, potentially more linear splines. Lower values allow more flexibility.
*   **Categorical Merge Method (`dropdown-xf-merge`):**
    *   How to handle categorical features after calculating their average effect (mean prediction) from the black-box model.
    *   `Quantile Binning`: Groups categories into a fixed number of bins based on the quantiles of their effects. The feature is replaced by the bin number.
    *   `None`: Replaces each category with its calculated average effect value (similar to target encoding using model predictions).
*   **Num. Quantile Bins (`input-n-bins`):** 
    *   Used only when 'Quantile Binning' is selected. Sets the number of groups to merge categories into.
    *   *Usage:* Controls the granularity of the categorical transformation.
*   **Keep Original Features (`switch-keep-original`):**
    *   If enabled, the original numeric and categorical features are kept in the output alongside their transformed versions (which will have `_xs` or `_xf` suffixes).
    *   If disabled (default), the original features selected for transformation are replaced by their transformed values.

---

**3. XSpliner Per-Feature Parameters (Numeric Features)**

*Note: The current app version uses 'auto' for all numeric features for simplicity. Future versions could allow per-feature control via a table.*

*   **Monotonicity (`monotonicity_map`):**
    *   Constrains the fitted spline to be always increasing or decreasing.
    *   `auto`: Automatically selects increasing or decreasing based on which fits the PDP better (lower RMSE).
    *   `inc`: Forces an increasing spline.
    *   `dec`: Forces a decreasing spline.
    *   `none`: Allows the spline to be non-monotonic.
*   **Alter Mode (`alter_map`):**
    *   Determines if the spline transformation should be applied or if the original feature is sufficient.
    *   `auto`: Compares the fit of the best spline (respecting monotonicity) to a simple linear fit on the PDP. If the linear fit is better or equal (based on RMSE), the spline transformation is *skipped* for that feature.
    *   `always`: Always applies the spline transformation.
    *   `never`: Never applies the spline transformation (keeps the original feature).

""")

# --- App Layout ---
sidebar = dbc.Card(
    [
        html.H4("1. Upload Data", className="card-title"),
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select CSV File')]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed',
                'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0'
            },
            multiple=False
        ),
        html.Hr(),
        html.H4("2. Select Features", className="card-title"),
        dbc.Label("Target Variable:", html_for='dropdown-target'),
        dcc.Dropdown(id='dropdown-target', placeholder="Select target...", className="mb-2"),
        dbc.Label("Numeric Features:", html_for='dropdown-numeric'),
        dcc.Dropdown(id='dropdown-numeric', placeholder="Select numeric...", multi=True, className="mb-2"),
        dbc.Label("Categorical Features:", html_for='dropdown-categorical'),
        dcc.Dropdown(id='dropdown-categorical', placeholder="Select categorical...", multi=True, className="mb-2"),
        html.Hr(),
        html.H4("2a. (Optional) Filter by Group", className="card-title text-muted"),
        dbc.Label("Grouping Column:", html_for='dropdown-group-col'),
        dcc.Dropdown(id='dropdown-group-col', placeholder="Select grouping column...", className="mb-2"),
        dbc.Label("Select Group Value:", html_for='dropdown-group-filter'),
        dcc.Dropdown(id='dropdown-group-filter', placeholder="Select group to analyze...", className="mb-2"),
        html.Hr(),
        # Add Date Range Selection Section
        html.H4("2b. Define Train/Test Periods", className="card-title"),
        dbc.Row([
            dbc.Col(dbc.Label("Train Start:"), width=6),
            dbc.Col(dbc.Label("Train End:"), width=6),
        ], className="mb-1"),
        dbc.Row([
             dbc.Col(dcc.DatePickerSingle(id='date-train-start', display_format='YYYY-MM-DD', placeholder='Train Start', className="mb-2", style={'width': '100%'}), width=6),
             dbc.Col(dcc.DatePickerSingle(id='date-train-end', display_format='YYYY-MM-DD', placeholder='Train End', className="mb-2", style={'width': '100%'}), width=6),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("Test Start:"), width=6),
            dbc.Col(dbc.Label("Test End:"), width=6),
        ], className="mb-1"),
         dbc.Row([
             dbc.Col(dcc.DatePickerSingle(id='date-test-start', display_format='YYYY-MM-DD', placeholder='Test Start', className="mb-2", style={'width': '100%'}), width=6),
             dbc.Col(dcc.DatePickerSingle(id='date-test-end', display_format='YYYY-MM-DD', placeholder='Test End', className="mb-2", style={'width': '100%'}), width=6),
        ], className="mb-2"),
        html.Hr(),
        html.H4("3. Select Models", className="card-title"),
        dbc.Label("Black-Box Model:", html_for='dropdown-blackbox'),
        dcc.Dropdown(
            id='dropdown-blackbox',
            options=[
                {'label': 'Random Forest Regressor', 'value': 'RandomForestRegressor'},
                {'label': 'Gradient Boosting Regressor', 'value': 'GradientBoostingRegressor'},
            ],
            value='RandomForestRegressor', className="mb-2"
        ),
        create_parameter_input("Num. Estimators:", "input-n-estimators", "Number of trees in the forest.", 100, min=10, step=10),
        dbc.Label("Final Interpretable Model:", html_for='dropdown-final'),
        dcc.Dropdown(
            id='dropdown-final',
            options=[
                {'label': 'Linear Regression', 'value': 'LinearRegression'},
                {'label': 'Ridge Regression', 'value': 'Ridge'},
            ],
            value='LinearRegression', className="mb-2"
        ),
        html.Hr(),
        html.H4("4. XSpliner Configuration", className="card-title"),
        html.P("Numeric Features (Monotonicity/Alter): Using 'auto' for all initially.", className="text-muted small"),
        # TODO: Add dynamic table/controls for per-feature settings if desired later
        html.H5("Global Parameters", className="mt-3"),
        create_parameter_input("PDP Grid Resolution:", "input-grid-res", "Number of points for PDP calculation.", 20, min=5, step=1),
        create_parameter_input("Num. Splines (GAM):", "input-n-splines", "Number of basis functions for GAM splines.", 8, min=2, step=1),
        create_parameter_input("Smoothing (Lambda):", "input-lam", "Smoothing penalty strength for GAM.", 0.6, min=0, step=0.1),
        dbc.Label("Categorical Merge Method:", html_for='dropdown-xf-merge'),
        dcc.Dropdown(
            id='dropdown-xf-merge',
            options=[{'label': 'Quantile Binning', 'value': 'quantile'}, {'label': 'None (Use Raw Effects)', 'value': 'none'}],
            value='quantile', className="mb-2"
        ),
        create_parameter_input("Num. Quantile Bins:", "input-n-bins", "Number of bins for quantile merging.", 5, min=2, step=1),
        dbc.Label("Keep Original Features:", html_for='switch-keep-original', className="mt-2"),
        dbc.Switch(id='switch-keep-original', value=False, label="Keep original alongside transformed", className="mb-3"),

        html.Hr(),
        dbc.Button("Run XSpliner Analysis", id="button-run", color="primary", className="w-100 mb-3", n_clicks=0),
        dbc.Alert(id='alert-error', color="danger", is_open=False),
        html.Div(id='div-status'),
    ],
    body=True,
)

# Define the main area with tabs
main_area_with_tabs = dbc.Tabs(
    [
        dbc.Tab(label="Analysis Results", children=[
            dbc.Card( # Keep the card styling for this tab's content
                [
                    html.H4("Results", className="card-title"),
                    html.Div(id='div-metrics-table'),
                    html.Hr(),
                    html.H5("Transformed Data Head"),
                    dash_table.DataTable(
                        id='table-transformed-head',
                        page_size=5,
                        style_table={'overflowX': 'auto'},
                        style_cell={
                            'height': 'auto',
                            'minWidth': '80px', 'width': '120px', 'maxWidth': '150px',
                            'whiteSpace': 'normal',
                            'textAlign': 'left',
                            'padding': '4px',
                            'fontSize': '12px'
                        },
                        style_header={
                            'fontWeight': 'bold',
                            'padding': '4px',
                            'fontSize': '13px'
                        }
                    ),
                    html.Hr(),
                    html.H5("Plots"), # Changed title slightly
                    html.Div(id='div-plots'),
                ],
                body=True,
                className="mt-3" # Add some margin top to separate from tabs
            )
        ]),
        dbc.Tab(label="Parameter Reference", children=[
             dbc.Card(dbc.CardBody(reference_content), className="mt-3")
        ]),
    ]
)

app.layout = dbc.Container(
    [
        dbc.Row(dbc.Col(html.H1("XSpliner Interactive Dashboard"), width=12), className="mb-4 mt-2"),
        dbc.Row(
            [
                dbc.Col(sidebar, width=12, lg=4, className="mb-4"),
                # Place the loading indicator around the tabs
                dbc.Col(dcc.Loading(id="loading-output", type="default", children=main_area_with_tabs), width=12, lg=8),
            ]
        ),
        dcc.Store(id='store-data'), # To store the dataframe
        dcc.Store(id='store-transformer-info'), # To store fitted transformer details
    ],
    fluid=True,
)

# --- Callbacks ---

@callback(
    [Output('dropdown-target', 'options'),
     Output('dropdown-target', 'value'),
     Output('dropdown-numeric', 'options'),
     Output('dropdown-numeric', 'value'),
     Output('dropdown-categorical', 'options'),
     Output('dropdown-categorical', 'value'),
     Output('dropdown-group-col', 'options'),
     Output('dropdown-group-col', 'value'),
     Output('dropdown-group-filter', 'options'),
     Output('dropdown-group-filter', 'value'),
     Output('date-train-start', 'min_date_allowed'),
     Output('date-train-start', 'max_date_allowed'),
     Output('date-train-start', 'initial_visible_month'),
     Output('date-train-start', 'date'),
     Output('date-train-end', 'min_date_allowed'),
     Output('date-train-end', 'max_date_allowed'),
     Output('date-train-end', 'initial_visible_month'),
     Output('date-train-end', 'date'),
     Output('date-test-start', 'min_date_allowed'),
     Output('date-test-start', 'max_date_allowed'),
     Output('date-test-start', 'initial_visible_month'),
     Output('date-test-start', 'date'),
     Output('date-test-end', 'min_date_allowed'),
     Output('date-test-end', 'max_date_allowed'),
     Output('date-test-end', 'initial_visible_month'),
     Output('date-test-end', 'date'),
     Output('store-data', 'data'),
     Output('alert-error', 'children', allow_duplicate=True),
     Output('alert-error', 'is_open', allow_duplicate=True)],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')],
    prevent_initial_call=True
)
def update_on_upload(contents, filename):
    """Parses uploaded CSV, updates feature selectors, and sets up date pickers."""
    # Initialize all outputs, including new date picker ones (24 date fields + 3 others)
    initial_outputs = ([]) * 10 + ([None] * 16) + [None, None, False] 
    if contents is None:
        return initial_outputs

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume comma-separated for simplicity
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return initial_outputs[:26] + [None, f"Invalid file type: {filename}", True]

        if df.empty:
             return initial_outputs[:26] + [None, "Uploaded CSV file is empty", True]

        all_cols = df.columns.tolist()
        date_col_name = None
        min_date, max_date = None, None
        # Find and parse date column, get min/max dates
        if 'date' in all_cols:
            date_col_name = 'date'
            try:
                 df[date_col_name] = pd.to_datetime(df[date_col_name])
                 min_date = df[date_col_name].min().date()
                 max_date = df[date_col_name].max().date()
            except Exception:
                 return initial_outputs[:26] + [None, f"Could not parse the 'date' column.", True]
        else:
             print("Warning: 'date' column not found for date picker setup.")
             # Proceed without setting date pickers - validation will fail later if needed
             min_date, max_date = pd.Timestamp('1900-01-01').date(), pd.Timestamp('2100-01-01').date() # Placeholder bounds

        # Identify other columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        # Exclude date column if found
        other_cols = [col for col in df.columns if col != date_col_name]
        categorical_cols = df[other_cols].select_dtypes(exclude=np.number).columns.tolist()
        
        # Ensure target is not accidentally excluded if it's numeric
        if 'target' in all_cols and 'target' not in numeric_cols and 'target' not in categorical_cols:
             # If target was the date column or something weird, handle appropriately
             # For now, just ensure numeric list is accurate if target is numeric
             if 'target' in df.select_dtypes(include=np.number).columns:
                   numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        # Prepare options for dropdowns (excluding date)
        target_options = [{'label': col, 'value': col} for col in all_cols if col != date_col_name]
        num_options = [{'label': col, 'value': col} for col in numeric_cols]
        cat_options = [{'label': col, 'value': col} for col in categorical_cols]
        # Grouping columns: Offer all non-target, non-date columns (could refine this)
        group_col_options = [
            {'label': col, 'value': col} for col in all_cols 
            if col != date_col_name 
               # Exclude potential default target initially? May need adjustment based on target selection.
               # and col != default_target 
        ]

        # Default target selection (excluding date)
        default_target = None
        potential_targets = [col for col in all_cols if col != date_col_name]
        if 'target' in potential_targets:
            default_target = 'target'
        elif numeric_cols: # Prioritize numeric cols for default target if 'target' not present
            default_target = numeric_cols[0]
        elif potential_targets: # Fallback to first non-date column
             default_target = potential_targets[0]

        # Default features (excluding date and target)
        default_numeric = [col for col in numeric_cols if col != default_target]
        default_categorical = [col for col in categorical_cols if col != default_target]

        # Calculate default date ranges (e.g., last 1 year for test)
        default_train_start = min_date
        default_test_end = max_date
        # Ensure calculation doesn't fail if max_date is None
        if max_date:
            # Use 1 year offset for default split
            default_train_end = (max_date - pd.DateOffset(years=1)).date() 
             # Handle case where data span is less than 1 year
            if default_train_end < min_date:
                 # If less than 1 year total, maybe split 70/30? Or just use first few points for train?
                 # Let's make train period at least 1 month if possible, else split roughly half
                 if (max_date - min_date).days > 60: 
                      default_train_end = (max_date - pd.DateOffset(months=int(0.3*12)+1)).date() # Approx 30% test
                      if default_train_end < min_date: default_train_end = min_date + pd.DateOffset(months=1) # Min 1 month train
                 else: # Very short data
                     default_train_end = min_date + pd.DateOffset(days=int(len(df)/2)) 
                 # Ensure train_end doesn't exceed max_date after adjustments
                 default_train_end = min(default_train_end, max_date - pd.DateOffset(days=1)) # Ensure at least 1 day for test
            default_test_start = (default_train_end + pd.DateOffset(days=1)).date()
            # Ensure test start doesn't exceed max date
            if default_test_start > max_date:
                 default_test_start = max_date
        else:
            default_train_end = None
            default_test_start = None
            
        # Date picker bounds and initial values
        min_dt_allowed = min_date
        max_dt_allowed = max_date
        initial_month = min_date or pd.Timestamp('now').date()

        # Return values including new date picker settings
        return (
            target_options, default_target,
            num_options, default_numeric,
            cat_options, default_categorical,
            group_col_options, None, # Group col options
            [], None, # Clear group filter 
            # Date Picker Outputs (bounds, initial month, default dates)
            min_dt_allowed, max_dt_allowed, initial_month, default_train_start, 
            min_dt_allowed, max_dt_allowed, initial_month, default_train_end,
            min_dt_allowed, max_dt_allowed, initial_month, default_test_start,
            min_dt_allowed, max_dt_allowed, initial_month, default_test_end,
            # Store and Alert
            df.to_dict('records'), 
            None, False
        )

    except Exception as e:
        print(traceback.format_exc())
        error_outputs = ([]) * 10 + ([None] * 16) + [None, f"Error processing file: {e}", True]
        return error_outputs

@callback(
    [Output('dropdown-group-filter', 'options', allow_duplicate=True),
     Output('dropdown-group-filter', 'value', allow_duplicate=True)],
    [Input('dropdown-group-col', 'value')],
    [State('store-data', 'data')],
    prevent_initial_call=True
)
def update_group_filter_options(group_col, data_records):
    """Updates the group filter dropdown options based on the selected group column."""
    if not group_col or not data_records:
        return [], None # No column selected or no data

    try:
        df = pd.DataFrame(data_records)
        if group_col not in df.columns:
            return [], None # Selected column not in data somehow
            
        unique_values = df[group_col].unique()
        unique_values.sort() # Sort for consistency
        
        options = [{'label': str(val), 'value': val} for val in unique_values]
        default_value = options[0]['value'] if options else None
        
        return options, default_value
        
    except Exception as e:
        print(f"Error updating group filter options: {e}")
        return [], None

@callback(
    [Output('div-metrics-table', 'children'),
     Output('table-transformed-head', 'data'),
     Output('table-transformed-head', 'columns'),
     Output('div-plots', 'children'),
     Output('div-status', 'children'),
     Output('store-transformer-info', 'data'),
     Output('alert-error', 'children', allow_duplicate=True),
     Output('alert-error', 'is_open', allow_duplicate=True)],
    [Input('button-run', 'n_clicks')],
    [State('date-train-start', 'date'),
     State('date-train-end', 'date'),
     State('date-test-start', 'date'),
     State('date-test-end', 'date'),
     State('store-data', 'data'),
     State('dropdown-target', 'value'),
     State('dropdown-numeric', 'value'),
     State('dropdown-categorical', 'value'),
     State('dropdown-blackbox', 'value'),
     State('input-n-estimators', 'value'),
     State('dropdown-final', 'value'),
     State('input-grid-res', 'value'),
     State('input-n-splines', 'value'),
     State('input-lam', 'value'),
     State('dropdown-xf-merge', 'value'),
     State('input-n-bins', 'value'),
     State('switch-keep-original', 'value'),
     State('dropdown-group-col', 'value'),
     State('dropdown-group-filter', 'value')],
     prevent_initial_call=True
)
def run_analysis(n_clicks, 
                 # Date picker values
                 train_start_date_str, train_end_date_str, test_start_date_str, test_end_date_str,
                 # Other state values
                 data_records, target_col, numeric_cols_selected, categorical_cols_selected,
                 bb_model_name, n_estimators, final_model_name,
                 grid_res, n_splines, lam, xf_merge_method, n_bins, keep_original,
                 group_col, group_value):
    """Runs the full analysis workflow using user-defined date ranges."""
    if not data_records or not target_col:
         return None, [], [], [], "Please upload data and select target variable.", None, None, False
    # Allow running without numeric features if categorical are selected? Maybe add check later.
    # if not numeric_cols_selected and not categorical_cols_selected:
    #      return None, [], [], [], "Please select at least one numeric or categorical feature.", None, None, False
         
    status_updates = ["Workflow started..."]
    error_msg = None
    error_open = False
    transformer_info = {}
    plots = []
    train_metrics_html = []
    test_metrics_html = []
    metrics_table_component = None
    table_data = []
    table_cols = []

    try:
        # Add check for selected dates
        if not all([train_start_date_str, train_end_date_str, test_start_date_str, test_end_date_str]):
             return None, [], [], [], "Please select all Train/Test start and end dates.", None, None, False
         
        df_full = pd.DataFrame(data_records) # Load full data first

        # --- 0. Filter Data by Group (if selected) ---
        if group_col and group_value is not None:
            if group_col not in df_full.columns:
                raise ValueError(f"Selected grouping column '{group_col}' not found in data.")
            status_updates.append(f"Filtering data for group: '{group_col}' = '{group_value}'.")
            df = df_full[df_full[group_col] == group_value].copy()
            if df.empty:
                raise ValueError(f"No data found for the selected group ('{group_col}' = '{group_value}').")
            status_updates.append(f"Filtered data shape: {df.shape}")
        else:
            df = df_full # Use the full dataframe if no filter applied
            status_updates.append("No group filter applied.")

        # --- 1. Data Preparation & Date Validation ---
        status_updates.append("Validating dates and preparing data...")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")
        if 'date' not in df.columns:
             raise ValueError("'date' column required for sequential time series split.")
             
        # Ensure date column is datetime and sort
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception as date_err:
             raise ValueError(f"Could not parse 'date' column: {date_err}")
        df = df.sort_values('date')
        
        # Convert selected date strings to datetime objects for comparison
        try:
            train_start_dt = pd.to_datetime(train_start_date_str).normalize()
            train_end_dt = pd.to_datetime(train_end_date_str).normalize()
            test_start_dt = pd.to_datetime(test_start_date_str).normalize()
            test_end_dt = pd.to_datetime(test_end_date_str).normalize()
        except Exception as date_parse_err:
             raise ValueError(f"Invalid date format selected: {date_parse_err}")

        # Validate date logic
        if train_start_dt > train_end_dt:
             raise ValueError("Train Start date cannot be after Train End date.")
        if test_start_dt > test_end_dt:
             raise ValueError("Test Start date cannot be after Test End date.")
        if train_end_dt >= test_start_dt:
             raise ValueError("Train End date must be before Test Start date.")
        
        # --- User-Defined Time Series Split ---
        status_updates.append("Performing user-defined train/test split...")
        train_mask = (df['date'] >= train_start_dt) & (df['date'] <= train_end_dt)
        test_mask = (df['date'] >= test_start_dt) & (df['date'] <= test_end_dt)
        
        if not train_mask.any() or not test_mask.any():
            raise ValueError("No data found within the specified Train or Test date ranges.")

        # Split using the masks on the potentially filtered df
        y = df[target_col]
        X_all = df.drop(columns=[target_col])
        X_all_train = X_all[train_mask]
        X_all_test = X_all[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]

        # Store original dates from the *split* data
        X_train_dates = X_all_train['date'].copy()
        X_test_original_dates = X_all_test['date'].copy()
        status_updates.append(f"Data split: {len(X_all_train)} train ({train_start_dt.date()} to {train_end_dt.date()}), "
                              f"{len(X_all_test)} test ({test_start_dt.date()} to {test_end_dt.date()}).")

        # Define selected features (numeric and categorical, EXCLUDING date)
        selected_numeric = numeric_cols_selected if numeric_cols_selected else []
        selected_categorical = categorical_cols_selected if categorical_cols_selected else []
        selected_features_for_modeling = selected_numeric + selected_categorical
        
        if not selected_features_for_modeling:
             raise ValueError("No numeric or categorical features selected for modeling.")

        # Now create X_train and X_test with only the selected features for modeling
        X_train = X_all_train[selected_features_for_modeling]
        X_test = X_all_test[selected_features_for_modeling]
        status_updates.append(f"Using features for modeling: {selected_features_for_modeling}")

        # --- 2. Black-Box Model ---
        status_updates.append(f"Training Black-Box Model ({bb_model_name})...")
        # Preprocessor needs the *selected* categorical columns list
        transformers = []
        if selected_categorical:
             transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), selected_categorical))
        # Only create preprocessor if there are transformers needed
        if transformers:
             preprocessor_rf = ColumnTransformer(transformers=transformers, remainder='passthrough')
        else:
             preprocessor_rf = 'passthrough' # No categorical features to transform

        # Define black-box model
        if bb_model_name == 'RandomForestRegressor':
            bb_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        elif bb_model_name == 'GradientBoostingRegressor':
             bb_model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
        else:
            raise ValueError(f"Unknown black-box model: {bb_model_name}")

        # Create pipeline
        black_box_pipeline = Pipeline(steps=[('preprocessor', preprocessor_rf), ('model', bb_model)])
        black_box_pipeline.fit(X_train, y_train)
        status_updates.append("Black-box model trained.")

        # --- 3. XSpliner Transformer ---
        status_updates.append("Configuring and fitting XSplinerTransformer...")
        # Transformer only needs the selected numeric/categorical features
        alter_map = {feature: 'auto' for feature in selected_numeric}
        monotonicity_map = {feature: 'auto' for feature in selected_numeric}

        pdp_params = {'grid_resolution': grid_res}
        gam_params = {'n_splines': n_splines, 's_kwargs': {'lam': lam}}
        xf_params = {'merge_method': xf_merge_method, 'merge_args': {'n_bins': n_bins}}

        xspliner = XSplinerTransformer(
            model=black_box_pipeline,
            numeric_features=selected_numeric,
            categorical_features=selected_categorical, # Date is no longer here
            monotonicity_map=monotonicity_map,
            alter_map=alter_map,
            compare_stat='rmse',
            pdp_params=pdp_params,
            gam_params=gam_params,
            xf_params=xf_params,
            keep_original=keep_original
        )
        # Fit transformer on X_train (which contains only selected features)
        xspliner.fit(X_train, y_train)
        status_updates.append("XSplinerTransformer fitted.")
        transformer_info = {
            'skipped_features': list(xspliner.skipped_features_),
            'categorical_mappings': {k: str(v) for k,v in xspliner.categorical_mappings_.items()} # Convert mapping dicts to strings for store
        }

        # --- 4. Transform Data ---
        status_updates.append("Transforming data...")
        X_train_transformed = xspliner.transform(X_train)
        X_test_transformed = xspliner.transform(X_test)

        # Prepare data for table display - ROUND numeric values
        X_train_transformed_head = X_train_transformed.head().copy()
        for col in X_train_transformed_head.select_dtypes(include=np.number).columns:
            X_train_transformed_head[col] = X_train_transformed_head[col].round(2)
            
        table_data = X_train_transformed_head.to_dict('records')
        table_cols = [{"name": i, "id": i} for i in X_train_transformed_head.columns]
        status_updates.append("Data transformed.")

        # --- 5. Final Model ---
        status_updates.append(f"Training Final Model ({final_model_name})...")
        if final_model_name == 'LinearRegression':
            final_model = LinearRegression()
        elif final_model_name == 'Ridge':
            final_model = Ridge(random_state=42)
        else:
             raise ValueError(f"Unknown final model: {final_model_name}")

        final_model.fit(X_train_transformed, y_train)
        status_updates.append("Final model trained.")

        # --- 6. Evaluation & Metric Calculation ---
        status_updates.append("Calculating Train & Test Metrics...")
        
        # Get predictions
        y_pred_train = final_model.predict(X_train_transformed)
        y_pred = final_model.predict(X_test_transformed)
        
        # Calculate metrics (store as numbers, not formatted strings)
        metrics_data = {
            'Metric': ['MSE', 'R2', 'MAE', 'MAPE (%)', 'SMAPE (%)'],
            'Train': [
                mean_squared_error(y_train, y_pred_train),
                r2_score(y_train, y_pred_train),
                mean_absolute_error(y_train, y_pred_train),
                mean_absolute_percentage_error(y_train, y_pred_train) * 100,
                symmetric_mean_absolute_percentage_error(y_train, y_pred_train)
            ],
            'Test': [
                mean_squared_error(y_test, y_pred),
                r2_score(y_test, y_pred),
                mean_absolute_error(y_test, y_pred),
                mean_absolute_percentage_error(y_test, y_pred) * 100,
                symmetric_mean_absolute_percentage_error(y_test, y_pred)
            ]
        }
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create DataTable for metrics with formatting
        metrics_table_component = dash_table.DataTable(
            columns=[
                {"name": "Metric", "id": "Metric"},
                {"name": "Train", "id": "Train", "type": "numeric", 
                 "format": Format(precision=2, scheme=Scheme.fixed) }, 
                {"name": "Test", "id": "Test", "type": "numeric", 
                 "format": Format(precision=2, scheme=Scheme.fixed) }
            ],
            data=metrics_df.to_dict('records'),
            style_table={'overflowX': 'auto', 'width': '80%', 'margin': 'auto'}, 
            style_cell={
                'padding': '5px',
                'fontSize': '13px',
                'textAlign': 'center' 
            },
            style_header={
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'Metric'},
                    'textAlign': 'left',
                    'fontWeight': 'bold' 
                }
            ]
        )
        status_updates.append("Metrics calculation complete.")

        # --- 7. Plotting ---
        status_updates.append("Generating plots...")
        plots = []
        # --- 7a. Actual vs Fitted/Predicted Time Series Plot ---
        status_updates.append("Generating Actual vs. Fitted/Predicted plot...")
        try:
            # Ensure we have the necessary date series from original df and split test dates
            if 'date' not in df.columns:
                 raise ValueError("Original DataFrame missing 'date' column for plotting.")
            if not isinstance(X_test_original_dates, pd.Series):
                 raise ValueError("X_test dates not available for plotting.")
                 
            fig_ts = go.Figure()

            # 1. Plot ALL actual values
            fig_ts.add_trace(go.Scatter(x=df['date'], y=df[target_col], # Use original df for full actuals
                                        mode='lines', name='Actual Target', line=dict(color='blue')))
            fig_ts.add_trace(go.Scatter(x=X_train_dates, y=y_pred_train, # Use stored train dates
                                        mode='lines', name='Fitted (Train)', line=dict(color='orange', dash='dot')))
            y_pred_series = pd.Series(y_pred, index=y_test.index) # Align test preds
            fig_ts.add_trace(go.Scatter(x=X_test_original_dates, y=y_pred_series, # Use stored test dates
                                        mode='lines', name='Predicted (Test)', line=dict(color='red', dash='dash')))
            fig_ts.add_vline(x=train_end_dt, line_width=2, line_dash="dash", line_color="grey")
            fig_ts.update_layout(
                title='Final Model: Actual vs. Fitted (Train) & Predicted (Test)',
                xaxis_title='Date',
                yaxis_title='Target Value',
                legend_title="Legend",
                margin=dict(l=20, r=20, t=40, b=20),
                height=450 # Increased height slightly
             )
            plots.append(dcc.Graph(id='graph-actual-vs-pred', figure=fig_ts))
        except Exception as ts_plot_err:
            print(f"Error generating time series plot: {ts_plot_err}")
            traceback.print_exc() # Print full traceback for debugging
            plots.append(dbc.Alert(f"Could not generate time series plot: {ts_plot_err}", color="warning"))

        # --- 7b. PDP & Fitted Spline Plots ---
        status_updates.append("Generating PDP/Spline plots...")
        features_to_plot = [f for f in selected_numeric if f in xspliner.fitted_gams_]
        if features_to_plot:
             # Pass X_train (includes selected numeric/categorical, excludes date) 
             # This matches the features the pipeline was fitted on.
             temp_xspliner = XSplinerTransformer(model=black_box_pipeline, numeric_features=features_to_plot, pdp_params=pdp_params)
             for feature in features_to_plot:
                 try:
                     gam = xspliner.fitted_gams_.get(feature)
                     if not gam: continue
                     
                     pdp_values, avg_responses = calculate_pdp(
                         temp_xspliner.model, 
                         X_train, # Use X_train (selected features only) for PDP calc
                         feature, 
                         **temp_xspliner.pdp_params 
                     )
                     if pdp_values is None or avg_responses is None: continue

                     # Generate points for GAM curve using the *actual fitted* GAM
                     gam_x = np.linspace(pdp_values.min(), pdp_values.max(), 100).reshape(-1, 1)
                     gam_y = gam.predict(gam_x)

                     fig = go.Figure()
                     fig.add_trace(go.Scatter(x=pdp_values, y=avg_responses, mode='markers', name='PDP Average Response', marker=dict(size=8)))
                     fig.add_trace(go.Scatter(x=gam_x.flatten(), y=gam_y, mode='lines', name='Fitted GAM Spline', line=dict(color='red', width=2)))
                     fig.update_layout(
                         title=f'PDP and Fitted Spline for: {feature}',
                         xaxis_title=feature,
                         yaxis_title='Partial Dependence (Target Scale)',
                         legend_title="Legend",
                         margin=dict(l=20, r=20, t=40, b=20),
                         height=400
                     )
                     plots.append(dcc.Graph(figure=fig))
                 except Exception as plot_err:
                      print(f"Error plotting feature {feature}: {plot_err}")
                      traceback.print_exc()
                      plots.append(dbc.Alert(f"Could not generate plot for {feature}: {plot_err}", color="warning"))
        else:
             plots.append(html.P("No numeric features were transformed with splines."))

        # Check if ONLY the time series plot was added (no splines fitted)
        if not features_to_plot and len(plots) > 0 and isinstance(plots[0], dcc.Graph) and plots[0].id == 'graph-actual-vs-pred':
             plots.append(html.P("No numeric features were transformed with splines."))
        elif not features_to_plot and len(plots) == 0: # If TS plot also failed
             plots.append(html.P("No plots generated."))

        status_updates.append("Workflow finished successfully.")

    except Exception as e:
        print(traceback.format_exc())
        error_msg = f"An error occurred: {e}"
        error_open = True
        status_updates.append("Workflow failed.")

    return (metrics_table_component, table_data, table_cols, plots,
            html.Ul([html.Li(s) for s in status_updates]),
            transformer_info, error_msg, error_open)


# --- Run App ---
if __name__ == '__main__':
    # Explicitly set the port number here
    app.run_server(debug=True, port=8053) 