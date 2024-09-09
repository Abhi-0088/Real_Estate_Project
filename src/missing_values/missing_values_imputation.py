import numpy as np
import pandas as pd

import os
import logging

logger = logging.getLogger('outlier_treatment')
logger.setLevel("DEBUG")

file_handler = logging.FileHandler('error.log')
file_handler.setLevel("ERROR")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

def load_data(data_path:str)->pd.DataFrame :
    try:
        datafrm = pd.read_csv(data_path)
        logger.debug("File Loaded Successfully from path :",data_path)

    except:
        logging.error("Error Occured while loading data using path :",data_path)
    return datafrm

def mode_based_imputation(df):
    for index,row in df.iterrows():
        if row['agePossession'] == 'Undefined':
            mode_value = df[(df['sector'] == row['sector']) & (df['property_type'] == row['property_type'])]['agePossession'].mode()
            # If mode_value is empty (no mode found), return NaN, otherwise return the mode
            if not mode_value.empty:
                row = mode_value.iloc[0] 
            else:
                row = np.nan
        else:
            continue
    return df['agePossession']

def mode_based_imputation2(df):
    for index,row in df.iterrows():
        if row['agePossession'] == 'Undefined':
            mode_value = df[(df['sector'] == row['sector'])]['agePossession'].mode()
            # If mode_value is empty (no mode found), return NaN, otherwise return the mode
            if not mode_value.empty:
                row =  mode_value.iloc[0] 
            else:
                row =  np.nan
        else:
            continue
    return df['agePossession']

def mode_based_imputation3(df):
    for index,row in df.iterrows():
        if row['agePossession'] == 'Undefined':
            mode_value = df[(df['property_type'] == row['property_type'])]['agePossession'].mode()
            # If mode_value is empty (no mode found), return NaN, otherwise return the mode
            if not mode_value.empty:
                row =  mode_value.iloc[0] 
            else:
                row =  np.nan
        else:
            continue
    return df['agePossession']


def missing_values_imputation(df:pd.DataFrame)->pd.DataFrame:

    all_present_df = df[~((df['super_built_up_area'].isnull()) | (df['built_up_area'].isnull()) | (df['carpet_area'].isnull()))]

    super_to_built_up_ratio = (all_present_df['super_built_up_area']/all_present_df['built_up_area']).median()
    carpet_to_built_up_ratio = (all_present_df['carpet_area']/all_present_df['built_up_area']).median()

    # both present built up null
    sbc_df = df[~(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & ~(df['carpet_area'].isnull())]

    sbc_df['built_up_area'].fillna(round(((sbc_df['super_built_up_area']/1.105) + (sbc_df['carpet_area']/0.9))/2),inplace=True)
    df.update(sbc_df)

    # sb present c is null built up null
    sb_df = df[~(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull())]

    sb_df['built_up_area'].fillna(round(sb_df['super_built_up_area']/1.105),inplace=True)
    df.update(sb_df)

    # sb null c is present built up null
    c_df = df[(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & ~(df['carpet_area'].isnull())]

    c_df['built_up_area'].fillna(round(c_df['carpet_area']/0.9),inplace=True)
    df.update(c_df)


    anamoly_df = df[(df['built_up_area'] < 2000) & (df['price'] > 2.5)][['price','area','built_up_area']]

    anamoly_df['built_up_area'] = anamoly_df['area']
    df.update(anamoly_df)

    df.drop(columns=['area','areaWithType','super_built_up_area','carpet_area','area_room_ratio'],inplace=True)

    ### floorNum


    df['floorNum'].fillna(2.0,inplace=True)

    ### facing

    df.drop(columns=['facing'],inplace=True)

    df.drop(index=[2536],inplace=True)

    ### agePossession



    df['agePossession'] = mode_based_imputation(df)


    df['agePossession'] = mode_based_imputation2(df)


    df['agePossession'] = mode_based_imputation3(df)

    return df

def save_data(final_df:pd.DataFrame,data_path:str)->None:

    data_path = os.path.join(data_path,'missing_value_imputation')
    os.makedirs(data_path,exist_ok=True)

    try:
        final_data_path = os.path.join(data_path,'missing_value_imputated_data.csv')
        final_df.to_csv(final_data_path,index=False)
    except:
        logging.error('Error Occured while saving data in path :',final_data_path)

def main():
    try:
        df = load_data(data_path = "Real_Estate_ML_Project/data/outliers_treated/outlier_treated_data.csv")
        final_df = missing_values_imputation(df)
        save_data(final_df,data_path="Real_Estate_ML_Project/data")
    except Exception as e:
        print(f"Error: {e}")
        logging.error("Error Occured in missing_value_imputation.py")

if __name__ == "__main__":
    main()

