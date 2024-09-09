import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
import shap



import os
import logging

logger = logging.getLogger('outlier_treatment')
logger.setLevel("DEBUG")

file_handler = logging.FileHandler('error.log')
file_handler.setLevel("ERROR")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def load_data(data_path:str)->pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logger.debug("Dataset Loaded Successfully")
        logging.error("File Loaded successfully in feature_selection.py")
    except:
        logging.error("Unable to load dataset from path: ",data_path)
    
    return df

def categorize_luxury(score):
    if 0 <= score < 50:
        return "Low"
    elif 50 <= score < 150:
        return "Medium"
    elif 150 <= score <= 175:
        return "High"
    else:
        return None  # or "Undefined"

def categorize_floor(floor):
    if 0 <= floor <= 2:
        return "Low Floor"
    elif 3 <= floor <= 10:
        return "Mid Floor"
    elif 11 <= floor <= 51:
        return "High Floor"
    else:
        return None  # or "Undefined" or any other label for floors outside the defined bins

def feature_selection(df:pd.DataFrame)->pd.DataFrame:

    train_df = df.drop(columns=['society','price_per_sqft'])

    ### luxury score
    # print(train_df.info())

    try:
        train_df['luxury_category'] = train_df['luxury_score'].apply(categorize_luxury)
        # print("printing train_df luxury_category-> " , train_df['luxury_category'])
    except Exception as e:
        print(f'Error {e}')
        print("Error occuerd in feature_selection_def")

    ### floorNum


    train_df['floor_category'] = train_df['floorNum'].apply(categorize_floor)

    train_df.drop(columns=['floorNum','luxury_score'],inplace=True)



    # Create a copy of the original data for label encoding
    data_label_encoded = train_df.copy()

    categorical_cols = train_df.select_dtypes(include=['object']).columns

    # Apply label encoding to categorical columns
    for col in categorical_cols:
        oe = OrdinalEncoder()
        data_label_encoded[col] = oe.fit_transform(data_label_encoded[[col]])
        # print(oe.categories_)

    # Splitting the dataset into training and testing sets
    X_label = data_label_encoded.drop('price', axis=1)
    y_label = data_label_encoded['price']

    ### Technique 1 - Correlation Analysis

    fi_df1 = data_label_encoded.corr()['price'].iloc[1:].to_frame().reset_index().rename(columns={'index':'feature','price':'corr_coeff'})

    ### Technique 2 - Random Forest Feature Importance


    # Train a Random Forest regressor on label encoded data
    rf_label = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_label.fit(X_label, y_label)

    # Extract feature importance scores for label encoded data
    fi_df2 = pd.DataFrame({
        'feature': X_label.columns,
        'rf_importance': rf_label.feature_importances_
    }).sort_values(by='rf_importance', ascending=False)


    ### Technique 3 - Gradient Boosting Feature importances


    # Train a Random Forest regressor on label encoded data
    gb_label = GradientBoostingRegressor()
    gb_label.fit(X_label, y_label)

    # Extract feature importance scores for label encoded data
    fi_df3 = pd.DataFrame({
        'feature': X_label.columns,
        'gb_importance': gb_label.feature_importances_
    }).sort_values(by='gb_importance', ascending=False)


    ### Technique 4 - Permutation Importance


    X_train_label, X_test_label, y_train_label, y_test_label = train_test_split(X_label, y_label, test_size=0.2, random_state=42)

    # Train a Random Forest regressor on label encoded data
    rf_label = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_label.fit(X_train_label, y_train_label)

    # Calculate Permutation Importance
    perm_importance = permutation_importance(rf_label, X_test_label, y_test_label, n_repeats=30, random_state=42)

    # Organize results into a DataFrame
    fi_df4 = pd.DataFrame({
        'feature': X_label.columns,
        'permutation_importance': perm_importance.importances_mean
    }).sort_values(by='permutation_importance', ascending=False)


    ### Technique 5 - LASSO


    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_label)

    # Train a LASSO regression model
    # We'll use a relatively small value for alpha (the regularization strength) for demonstration purposes
    lasso = Lasso(alpha=0.01, random_state=42)
    lasso.fit(X_scaled, y_label)

    # Extract coefficients
    fi_df5 = pd.DataFrame({
        'feature': X_label.columns,
        'lasso_coeff': lasso.coef_
    }).sort_values(by='lasso_coeff', ascending=False)



    ### Technique 6 - RFE


    # Initialize the base estimator
    estimator = RandomForestRegressor()

    # Apply RFE on the label-encoded and standardized training data
    selector_label = RFE(estimator, n_features_to_select=X_label.shape[1], step=1)
    selector_label = selector_label.fit(X_label, y_label)

    # Get the selected features based on RFE
    selected_features = X_label.columns[selector_label.support_]

    # Extract the coefficients for the selected features from the underlying linear regression model
    selected_coefficients = selector_label.estimator_.feature_importances_

    # Organize the results into a DataFrame
    fi_df6 = pd.DataFrame({
        'feature': selected_features,
        'rfe_score': selected_coefficients
    }).sort_values(by='rfe_score', ascending=False)



    ### Technique 7 - Linear Regression Weights

    # Train a linear regression model on the label-encoded and standardized training data
    lin_reg = LinearRegression()
    lin_reg.fit(X_scaled, y_label)

    # Extract coefficients
    fi_df7 = pd.DataFrame({
        'feature': X_label.columns,
        'reg_coeffs': lin_reg.coef_
    }).sort_values(by='reg_coeffs', ascending=False)



    ### Technique 8 - SHAP

    # Compute SHAP values using the trained Random Forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_label, y_label)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_label)

    # Summing the absolute SHAP values across all samples to get an overall measure of feature importance
    shap_sum = np.abs(shap_values).mean(axis=0)



    fi_df8 = pd.DataFrame({
        'feature': X_label.columns,
        'SHAP_score': np.abs(shap_values).mean(axis=0)
    }).sort_values(by='SHAP_score', ascending=False)


    final_fi_df = fi_df1.merge(fi_df2,on='feature').merge(fi_df3,on='feature').merge(fi_df4,on='feature').merge(fi_df5,on='feature').merge(fi_df6,on='feature').merge(fi_df7,on='feature').merge(fi_df8,on='feature').set_index('feature')


    # normalize the score
    final_fi_df = final_fi_df.divide(final_fi_df.sum(axis=0), axis=1)
    final_fi_df[['rf_importance','gb_importance','permutation_importance','rfe_score','SHAP_score']].mean(axis=1).sort_values(ascending=False)
    # to drop pooja room, study room, others

    # with all the cols


    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    # scores = cross_val_score(rf, X_label, y_label, cv=5, scoring='r2')
    # scores.mean()


    scores = cross_val_score(rf, X_label.drop(columns=['pooja room', 'study room', 'others']), y_label, cv=5, scoring='r2')
    logging.error("The baseline model score is : ", scores.mean())

    export_df = X_label.drop(columns=['pooja room', 'study room', 'others'])
    export_df['price'] = y_label

    ## making changes for ease in model_selection process

    export_df['luxury_category'] = train_df['luxury_category']
    export_df['floor_category'] = train_df['floor_category']

    export_df['property_type'] = train_df['property_type']
    export_df['sector'] = train_df['sector']

    export_df['balcony'] = train_df['balcony']
    export_df['agePossession'] = train_df['agePossession']


    

    return export_df

def save_data(final_df:pd.DataFrame,data_path:str)->None:
    data_path = os.path.join(data_path,"feature_selection")
    os.makedirs(data_path,exist_ok=True)

    try:
        final_data_path = os.path.join(data_path,'post_feature_selection_data.csv')
        final_df.to_csv(final_data_path,index=False)
    except:
        logging.error("Error occured while saving data in path :",final_data_path)

def main():
    try:
        df = load_data("Real_Estate_ML_Project/data/missing_value_imputation/missing_value_imputated_data.csv")
        final_df = feature_selection(df)
        save_data(final_df,data_path="Real_Estate_ML_Project/data")
    except Exception as e:
        print(f"Error: {e}")
        logger.error("Error Occured in Feature Selection.py")

if __name__ == "__main__":
    main()

