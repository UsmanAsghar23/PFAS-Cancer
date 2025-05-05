from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def predict_cancer_incidence(df):
    # Group by specified columns and calculate average PFAS concentration
    avg_df = df[df['Cancer'] == 'AllSite'].groupby(['county', 'Sex', 'PopTot', "Cancer_Incidents"])['total_pfas_concentration'].mean().reset_index()
    avg_df = avg_df[["PopTot", "total_pfas_concentration", "Sex", "Cancer_Incidents"]]
    avg_df['Sex'] = (avg_df['Sex'] == 'Female').astype(int)

    features = ['total_pfas_concentration', 'PopTot', 'Sex']
    target = 'Cancer_Incidents'

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        avg_df[features], 
        avg_df[target], 
        test_size=0.2, 
        random_state=42
    )

    # Baseline: mean
    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_rmse = mean_squared_error(y_test, baseline_pred)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_rmse = mean_squared_error(y_test, y_pred_rf)

    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    gb_rmse = mean_squared_error(y_test, y_pred_gb)


    return rf_rmse, gb_rmse, baseline_rmse

