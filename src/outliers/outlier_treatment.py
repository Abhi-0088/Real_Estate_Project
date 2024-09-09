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


def outlier_treatment(df:pd.DataFrame)->pd.DataFrame:

    # outliers on the basis of price column

    # Calculate the IQR for the 'price' column
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR


    # Calculate the IQR for the 'price' column
    Q1 = df['price_per_sqft'].quantile(0.25)
    Q3 = df['price_per_sqft'].quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers_sqft = df[(df['price_per_sqft'] < lower_bound) | (df['price_per_sqft'] > upper_bound)]

    outliers_sqft['area'] = outliers_sqft['area'].apply(lambda x:x*9 if x<1000 else x)
    outliers_sqft['price_per_sqft'] = round((outliers_sqft['price']*10000000)/outliers_sqft['area'])

    df.update(outliers_sqft)


    df = df[df['price_per_sqft'] <= 50000]




    df = df[df['area'] < 100000]



    # 818, 1796, 1123, 2, 2356, 115, 3649, 2503, 1471
    df.drop(index=[818, 1796, 1123, 2, 2356, 115, 3649, 2503, 1471], inplace=True)

    df.loc[48,'area'] = 115*9
    df.loc[300,'area'] = 7250
    df.loc[2666,'area'] = 5800
    df.loc[1358,'area'] = 2660
    df.loc[3195,'area'] = 2850
    df.loc[2131,'area'] = 1812
    df.loc[3088,'area'] = 2160
    df.loc[3444,'area'] = 1175

    ### Bedroom


    df = df[df['bedRoom'] <= 10]




    ### built up area

    df.loc[2131,'carpet_area'] = 1812

    df['price_per_sqft'] = round((df['price']*10000000)/df['area'])

    x = df[df['price_per_sqft'] <= 20000]


    df['area_room_ratio'] = df['area']/df['bedRoom']

    outliers_df = df[(df['area_room_ratio'] < 250) & (df['bedRoom'] > 3)]
    outliers_df['bedRoom'] = round(outliers_df['bedRoom']/outliers_df['floorNum'])
    df.update(outliers_df)
    df['area_room_ratio'] = df['area']/df['bedRoom']

    df = df[~((df['area_room_ratio'] < 250) & (df['bedRoom']>4))]

    return df


def save_data(final_df:pd.DataFrame,data_path:str)->None:
    data_path = os.path.join(data_path,'outliers_treated')
    os.makedirs(data_path,exist_ok=True)
    try:
        final_data_path = os.path.join(data_path,'outlier_treated_data.csv')
        final_df.to_csv(final_data_path,index=False)
    except:
        logging.error("Unable to store final df in path :",final_data_path)


def main():
    try:
        df = load_data(data_path='Real_Estate_ML_Project/data/featured_data/preprocessed_dataset_v2.csv')
        fianl_df = outlier_treatment(df)
        save_data(fianl_df,data_path="Real_Estate_ML_Project/data")
    except:
        logging.error("Error Occured in outlier_treatment.py")


if __name__ == "__main__":
    main()


