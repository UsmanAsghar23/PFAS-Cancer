import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve


def prepare_data(df):
    numeric_df = df.select_dtypes(include='number')
    correlations = numeric_df.corr()['AAIR'].dropna()
    exclude = ['total_pfas_concentration', 'PopTot', 'Cancer_Incidents', 'AAIR']
    chemical_correlations = correlations.drop(labels=exclude, errors='ignore')
    top_10_chemicals = chemical_correlations.abs().sort_values(ascending=False).head(10)
    top_10_corr = chemical_correlations.loc[top_10_chemicals.index]
    top_10_chemicals_list = top_10_corr.index.tolist()

    non_chemical_columns = [
        'county',
        'gm_samp_collection_date',
        'total_pfas_concentration',
        'Cancer_Incidents',
        'AAIR',
        'PopTot',
        'Sex',
        'Cancer'
    ]

    reduced_df = df[top_10_chemicals_list + non_chemical_columns]

    encoded_df = pd.get_dummies(reduced_df, columns=['Sex', 'Cancer'], drop_first=True)

    # Create AAIR binary label
    mean_aair = encoded_df['AAIR'].mean()
    encoded_df['AAIR_Label'] = (encoded_df['AAIR'] > mean_aair).astype(int)

    # Save the encoded dataframe with all features
    encoded_df.to_csv("reduced_pfas_dataset.csv", index=False)

    # Prepare features and labels for ML
    features_to_drop = ['AAIR', 'Cancer_Incidents', 'gm_samp_collection_date', 'county']
    X = encoded_df.drop(columns=features_to_drop + ['AAIR_Label'])
    y = encoded_df['AAIR_Label']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
    return X_train, X_test, y_train, y_test

def train_logistic_regression_model(X_train, y_train, X_test, y_test):
    # Train Logistic Regression with L1 regularization
    log_reg = LogisticRegression(
        penalty='l1',          # L1 regularization
        solver='liblinear',    # solver that supports L1
        C=1.0,                 # regularization strength (smaller C = stronger regularization)
        random_state=50
    )

    log_reg.fit(X_train, y_train)

    # Predict
    y_pred = log_reg.predict(X_test)

    # Evaluation
    print("Logistic Regression (L1) Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Feature Importance (non-zero coefficients)
    coef = log_reg.coef_[0]
    feature_names = X_train.columns

    # Only show features that survived (non-zero)
    non_zero_features = feature_names[np.abs(coef) > 1e-6]
    non_zero_coefs = coef[np.abs(coef) > 1e-6]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=non_zero_coefs, y=non_zero_features, orient='h')
    plt.title("Feature Importances from L1 Logistic Regression")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    # Optional: Plot ROC Curve
    y_pred_proba = log_reg.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_proba):.2f}')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Logistic Regression with L1')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def train_random_forest_model(X_train, y_train, X_test, y_test):
    # train a Random Forest Classifier
    clf = RandomForestClassifier(
        random_state=50,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        n_estimators=100
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Model Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # plot 10 most important features
    importances = clf.feature_importances_
    feature_names = X_train.columns
    top_indices = np.argsort(importances)[-10:]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances[top_indices], y=np.array(feature_names)[top_indices])
    plt.title("Top 10 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def predict_cancer_incidence(df):
    """
    Predicts cancer incidence based on PFAS concentration and other features.
    """
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

    print(f"Random Forest RMSE: {rf_rmse**0.5:.3f}")
    print(f"Gradient Boosting RMSE: {gb_rmse**0.5:.3f}")
    print(f"Baseline RMSE: {baseline_rmse**0.5:.3f}")

    return rf_rmse, gb_rmse, baseline_rmse