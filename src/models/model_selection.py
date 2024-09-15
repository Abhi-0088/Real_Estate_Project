import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.decomposition import PCA

import category_encoders as ce

import pickle


import os
import logging

logger = logging.getLogger('model_selection')
logger.setLevel("DEBUG")

file_handler = logging.FileHandler('output_model_selection.log')
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def load_data(data_path:str)->pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logger.debug("Dataset Loaded Successfully")
    except:
        logging.error("Unable to load dataset from path: ",data_path)
    
    return df

def scorer(model_name, model,preprocessor,X,Y):
    
    output = []

    output.append(model_name)
    

    
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    try:
    # K-fold cross-validation
        kfold = KFold(n_splits=6, shuffle=True, random_state=23)
        scores = cross_val_score(pipeline, X, Y, cv=kfold, scoring='r2')
    
    except Exception as e:
        logging.error("Error in model_selection.py in cross val score-> ",e)

    
    output.append(scores.mean())

    
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
    
    pipeline.fit(X_train,y_train)
    
    y_pred = pipeline.predict(X_test)
    
    y_pred = np.expm1(y_pred)
    
    output.append(mean_absolute_error(np.expm1(y_test),y_pred))
    
    return output

def scorer_with_pca(model_name, model,preprocessor,X,Y):
    
    output = []
    
    output.append(model_name)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=0.95)),
        ('regressor', model)
    ])
    
    # K-fold cross-validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, Y, cv=kfold, scoring='r2')
    
    output.append(scores.mean())
    
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
    
    pipeline.fit(X_train,y_train)
    
    y_pred = pipeline.predict(X_test)
    
    y_pred = np.expm1(y_pred)
    
    output.append(mean_absolute_error(np.expm1(y_test),y_pred))
    
    return output



# 0 -> unfurnished
# 1 -> semifurnished
# 2 -> furnished
def model_selection(df:pd.DataFrame)->None:
    df['furnishing_type'] = df['furnishing_type'].replace({0.0:'unfurnished',1.0:'semifurnished',2.0:'furnished'})

    # dropping these values becuase there is only one record related to this sector type because of which ordinalencoding
    #throw error i.e we are removing these rows
    index_of_sector17a = df[df['sector'].str.contains("sector 17a")].index
    df.drop(index=index_of_sector17a,inplace=True)

    #same reason as above only one value of record containing sector 37 as sector type
    df.drop(index=893,inplace=True)

    X = df.drop(columns=['price'])
    y = df['price']
    # Applying the log1p transformation to the target variable
    y_transformed = np.log1p(y)
    ### Ordinal Encoding

    # Creating a column transformer for preprocessing
    columns_to_encode = ['property_type','sector', 'balcony', 'agePossession', 'furnishing_type', 'luxury_category', 'floor_category']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
            ('cat', OrdinalEncoder(), columns_to_encode)
        ], 
        remainder='passthrough'
    )
    # Creating a pipeline
    # pipeline = Pipeline([
    #     ('preprocessor', preprocessor),
    #     ('regressor', LinearRegression())
    # ])
    # K-fold cross-validation
    # kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    # scores = cross_val_score(pipeline, X, y_transformed, cv=kfold, scoring='r2')
    # # scores.mean(),scores.std()
    # X_train, X_test, y_train, y_test = train_test_split(X,y_transformed,test_size=0.2,random_state=42)
    # pipeline.fit(X_train,y_train)
    # y_pred = pipeline.predict(X_test)
    # y_pred = np.expm1(y_pred)

    # mean_absolute_error(np.expm1(y_test),y_pred)



        
    model_dict = {
        'linear_reg':LinearRegression(),
        'svr':SVR(),
        'ridge':Ridge(),
        'LASSO':Lasso(),
        'decision tree': DecisionTreeRegressor(),
        'random forest':RandomForestRegressor(),
        'extra trees': ExtraTreesRegressor(),
        'gradient boosting': GradientBoostingRegressor(),
        'adaboost': AdaBoostRegressor(),
        'mlp': MLPRegressor(),
        'xgboost':XGBRegressor()
    }

    model_output = []
    for model_name,model in model_dict.items():
        output = scorer(model_name, model,preprocessor,X,y_transformed)
        model_output.append(output)
        

    model_df = pd.DataFrame(model_output, columns=['name','r2','mae'])
    model_df.sort_values(['mae'])
    

    logger.debug('Output with Ordinal Encoding is ->\n')
    print('Output with Ordinal Encoding is ->\n',model_df)



    ### OneHotEncoding
    # Creating a column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
            ('cat', OrdinalEncoder(), columns_to_encode),
            ('cat1',OneHotEncoder(drop='first'),['sector','agePossession','furnishing_type'])
        ], 
        remainder='passthrough'
    )
    # # Creating a pipeline
    # pipeline = Pipeline([
    #     ('preprocessor', preprocessor),
    #     ('regressor', LinearRegression())
    # ])

    # X_train, X_test, y_train, y_test = train_test_split(X,y_transformed,test_size=0.2,random_state=42)
    # pipeline.fit(X_train,y_train)
    # y_pred = pipeline.predict(X_test)
    # y_pred = np.expm1(y_pred)
    # mean_absolute_error(np.expm1(y_test),y_pred)

        
    model_dict = {
        'linear_reg':LinearRegression(),
        'svr':SVR(),
        'ridge':Ridge(),
        'LASSO':Lasso(),
        'decision tree': DecisionTreeRegressor(),
        'random forest':RandomForestRegressor(),
        'extra trees': ExtraTreesRegressor(),
        'gradient boosting': GradientBoostingRegressor(),
        'adaboost': AdaBoostRegressor(),
        'mlp': MLPRegressor(),
        'xgboost':XGBRegressor()
    }
    model_output = []
    for model_name,model in model_dict.items():
        model_output.append(scorer(model_name, model,preprocessor,X,y_transformed))
    model_df = pd.DataFrame(model_output, columns=['name','r2','mae'])
    model_df.sort_values(['mae'])

    logging.debug("Ouput_after_one_hot_endocing->\n",model_df.to_string())
    print("Ouput_after_one_hot_endocing->\n",model_df)
    ### OneHotEncoding With PCA
    # Creating a column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
            ('cat', OrdinalEncoder(), columns_to_encode),
            ('cat1',OneHotEncoder(drop='first',sparse_output=False),['sector','agePossession'])
        ], 
        remainder='passthrough'
    )
    # Creating a pipeline
    # pipeline = Pipeline([
    #     ('preprocessor', preprocessor),
    #     ('pca', PCA(n_components=0.95)),
    #     ('regressor', LinearRegression())
    # ])


        
    model_dict = {
        'linear_reg':LinearRegression(),
        'svr':SVR(),
        'ridge':Ridge(),
        'LASSO':Lasso(),
        'decision tree': DecisionTreeRegressor(),
        'random forest':RandomForestRegressor(),
        'extra trees': ExtraTreesRegressor(),
        'gradient boosting': GradientBoostingRegressor(),
        'adaboost': AdaBoostRegressor(),
        'mlp': MLPRegressor(),
        'xgboost':XGBRegressor()
    }
    model_output = []
    for model_name,model in model_dict.items():
        model_output.append(scorer_with_pca(model_name, model,preprocessor,X,y_transformed))
    model_df = pd.DataFrame(model_output, columns=['name','r2','mae'])
    model_df.sort_values(['mae'])

    logging.debug("output using one-hot encoding and pca->\n",model_df.to_string())
    print("output using one-hot encoding and pca->\n",model_df)
    ### Target Encoder

    columns_to_encode = ['property_type','sector', 'balcony', 'agePossession', 'furnishing_type', 'luxury_category', 'floor_category']

    # Creating a column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
            ('cat', OrdinalEncoder(), columns_to_encode),
            ('cat1',OneHotEncoder(drop='first',sparse_output=False),['agePossession']),
            ('target_enc', ce.TargetEncoder(), ['sector'])
        ], 
        remainder='passthrough'
    )
    # !pip install category_encoders
    # Creating a pipeline

    # K-fold cross-validation

        
    model_dict = {
        'linear_reg':LinearRegression(),
        'svr':SVR(),
        'ridge':Ridge(),
        'LASSO':Lasso(),
        'decision tree': DecisionTreeRegressor(),
        'random forest':RandomForestRegressor(),
        'extra trees': ExtraTreesRegressor(),
        'gradient boosting': GradientBoostingRegressor(),
        'adaboost': AdaBoostRegressor(),
        'mlp': MLPRegressor(),
        'xgboost':XGBRegressor()
    }
    model_output = []
    for model_name,model in model_dict.items():
        model_output.append(scorer(model_name, model,preprocessor,X,y_transformed))
    model_df = pd.DataFrame(model_output, columns=['name','r2','mae'])
    model_df.sort_values(['mae'])

    logging.debug('output using category_encoders->\n',model_df.to_string())
    print('output using category_encoders->\n',model_df)


def final_model_selection(df:pd.DataFrame,pipeline_datapath:str,final_df_datapath:str)->None:
    # df['furnishing_type'] = df['furnishing_type'].replace({0.0:'unfurnished',1.0:'semifurnished',2.0:'furnished'})
    print("information of dataframe is->\n",df.info())

    # dropping these values becuase there is only one record related to this sector type because of which ordinalencoding
    #throw error i.e we are removing these rows
    # index_of_sector17a = df[df['sector'].str.contains("sector 17a")].index
    # df.drop(index=index_of_sector17a,inplace=True)

    # #same reason as above only one value of record containing sector 37 as sector type
    # df.drop(index=893,inplace=True)

    X = df.drop(columns=['price'])
    y = df['price']
    # Applying the log1p transformation to the target variable
    y_transformed = np.log1p(y)
    ### Ordinal Encoding

    # Creating a column transformer for preprocessing
    columns_to_encode = ['property_type','sector', 'balcony', 'agePossession', 'furnishing_type', 'luxury_category', 'floor_category']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
            ('cat', OrdinalEncoder(), columns_to_encode),
            ('cat1',OneHotEncoder(drop='first',sparse_output=False),['sector','agePossession'])
        ], 
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ('preprocessor',preprocessor),
        ('regressor',XGBRegressor(n_estimators=500))
    ])

    pipeline.fit(X,y_transformed)

    y_hat = np.expm1(pipeline.predict(X))

    mae = mean_absolute_error(y,y_hat)


    print("MAE of final_model is-> ", mae)
    logger.debug("MAE of final_model is-> ", str(mae))

    pipeline_datapath = os.path.join(pipeline_datapath,'model')
    os.makedirs(pipeline_datapath,exist_ok=True);

    final_df_datapath = os.path.join(final_df_datapath,'final_x_train')
    os.makedirs(final_df_datapath,exist_ok=True);

    pipeline_datapath = os.path.join(pipeline_datapath,"xg_boost_pipeline.pkl")
    final_x_train_datapath = os.path.join(final_df_datapath,"final_x_train.pkl")
    final_y_train_datapath = os.path.join(final_df_datapath,"final_y_train.pkl")


    with open(pipeline_datapath, 'wb') as file:
        pickle.dump(pipeline, file)
    
    with open(final_x_train_datapath,'wb') as file:
        pickle.dump(X,file)

    with open(final_y_train_datapath,'wb') as file:
        pickle.dump(y,file)

def main():
    try:
        df = load_data("data/feature_selection/post_feature_selection_data.csv")
        model_selection(df)
        final_model_selection(df,pipeline_datapath="C:/Users/abhil/OneDrive/Desktop/real_estate_project/Real_Estate_ML_Project",final_df_datapath="C:/Users/abhil/OneDrive/Desktop/real_estate_project/Real_Estate_ML_Project/data")
    except Exception as e:
        logger.error("Error occurd in model_selection.py-> ",e)
        print("Error occured in model_selection.py\n")


if __name__ == "__main__":
    main()


