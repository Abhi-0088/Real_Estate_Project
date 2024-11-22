import numpy as np
import pandas as pd
import pickle 
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
import json
import os
import logging
import yaml
import mlflow
import mlflow.sklearn
from typing import Tuple,Dict
import os

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Abhi-0088"
repo_name = "dagshub-demo"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s',params_path)
        return params
    except FileNotFoundError:
        logging.error("Error Occured in model evalutation ,showing file not found while loading params")
        print('File Not Found from {params_path}')
    except Exception as e:
        print("Error Occured in model evaluation while loading params")


def load_data(data_path:str)->pd.DataFrame:
    try:
        with open(data_path,'rb') as file:
            df = pickle.load(file)
        logger.debug('In model evaluation, dataset loaded successfully')
        return df
    except Exception as e:
        print(f'Error Occured in model evaluation {e}')


def train_model(params:dict) -> Pipeline:
    try:
        columns_to_encode = ['property_type','sector', 'balcony', 'agePossession', 'furnishing_type', 'luxury_category', 'floor_category']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
                ('cat', OrdinalEncoder(), columns_to_encode),
                ('cat1',OneHotEncoder(drop='first',sparse_output=False),['sector','agePossession'])
            ], 
            remainder='passthrough'
        )
        #print("n_estimator is ",params['n_estimators'])

        pipeline = Pipeline([
            ('preprocessor',preprocessor),
            ('regressor',XGBRegressor(n_estimators=params['n_estimators']))
        ])


        return pipeline

    except Exception as e:
        logging.error("Error occured in model training in model evaluation")
        print("Error Occured in model evaluation",e)

def evaluate_model(X_train:pd.DataFrame,y_train:pd.DataFrame,pipeline,params:dict,final_X_test_path:str)->Tuple[Dict,Pipeline]:
    try:
        x_train,x_test,Y_train,Y_test = train_test_split(X_train,y_train,test_size=params['test_size'],random_state=42)

        y_train_transformed = np.log1p(Y_train)
        fitted_pipeline = pipeline.fit(x_train,y_train_transformed)

        y_hat = np.expm1(pipeline.predict(x_test))

        mae = mean_absolute_error(Y_test,y_hat)
        r2_scr = r2_score(Y_test,y_hat)

        metrics_dict = {
            'r2_score' : r2_scr,
            'mae' : mae
        }

        final_test_path = os.path.join(final_X_test_path,"final_test_data")
        os.makedirs(final_test_path,exist_ok=True)

        final_x_test_path = os.path.join(final_test_path,"final_X_test.pkl")
        final_y_test_path = os.path.join(final_test_path,"final_Y_test.pkl")

        with open(final_x_test_path,'wb') as x_test_file:
            pickle.dump(x_test,x_test_file)

        with open(final_y_test_path,'wb') as y_test_file:
            pickle.dump(Y_test,y_test_file)

        logger.debug('model evaluation metrics calculated'  )
        return metrics_dict,fitted_pipeline

    except Exception as e:
        logger.error("Error occured in model evalutation")
        print("Error occured in evaluate_model in model evalutaion",e)

def save_metrics(metrics:dict, file_path:str) ->None:
    try:
        with open(file_path,'w') as file:
            json.dump(metrics,file,indent=4)
        #logger.debug("Metrics saved to %s",file_path)
    except Exception as e:
        logger.error("Failed to complete the save_metrics process")
        print("Error occured in model_evaluation inside save_metrics",e)

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    mlflow.set_experiment("real_esate_project_expr")
    with mlflow.start_run() as run:

        try:
            X_train = load_data('data/final_x_train/final_x_train.pkl')
            y_train = load_data('data/final_x_train/final_y_train.pkl')

            params = load_params('params.yaml')['model_evaluation']

            pipeline = train_model(params)

            metrics_dict,fitted_pipeline = evaluate_model(X_train,y_train,pipeline,params,final_X_test_path="data/")

            save_metrics(metrics_dict,file_path='model/model_metric.json')

            

            for metric_name,metric_value in metrics_dict.items():
                mlflow.log_metric(metric_name,metric_value)
            
            logger.debug("Metrics logged successfuly")    


            print(type(params))
            for params_name,params_value in params.items():
                mlflow.log_param(params_name,params_value)
            
            logger.debug("Params logged Successfuly")

  
            
            mlflow.sklearn.log_model(fitted_pipeline,"model")

            save_model_info(run.info.run_id,"model",'model/experiments_info.json')

            mlflow.log_artifact('model/model_metric.json')

            mlflow.log_artifact('model/experiments_info.json')    

            logger.debug("Model Evaluation Completed Sucessfuly")        
            

            
        except Exception as e:
            print("Error occured in model_evaluatin.py",e)

if __name__ == "__main__":
    main()


