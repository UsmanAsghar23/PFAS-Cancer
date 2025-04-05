from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_ind
import pandas as pd
import numpy as np

def predict_cancer_incidence(df):
    # Group by specified columns and calculate average PFAS concentration
    avg_df = df.groupby(['county', 'Cancer', 'Sex', 'Cancer_Incidents'])['total_pfas_concentration'].mean().reset_index()
    
    features = ['total_pfas_concentration', 'county', 'Cancer', 'Sex']
    target = 'Cancer_Incidents'

    # Convert categorical features to numerical using one-hot encoding
    avg_df = pd.get_dummies(avg_df, columns=['county', 'Cancer', 'Sex'])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        avg_df[features], 
        avg_df[target], 
        test_size=0.2, 
        random_state=42
    )

    # Baseline: mean
    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_rmse = mean_squared_error(y_test, baseline_pred, squared=False)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_rmse = mean_squared_error(y_test, y_pred_rf, squared=False)

    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    gb_rmse = mean_squared_error(y_test, y_pred_gb, squared=False)


    return rf_rmse, gb_rmse, baseline_rmse

