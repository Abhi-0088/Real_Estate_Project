# load test + signature test + performance test

import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import  mean_absolute_error,r2_score
import pickle

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "Abhi"
        repo_name = "dagshub-demo"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load the new model from MLflow model registry
        cls.new_model_name = "XGBoost_Model_1"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        # # Load the vectorizer
        # cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

        # Load holdout test data
        with open('data/final_test_data/final_X_test.pkl','rb') as x_test:
            cls.holdout_x_data = pickle.load(x_test)

        with open('data/final_test_data/final_Y_test.pkl','rb') as y_test:
            cls.holdout_y_data = pickle.load(y_test)

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    # def test_model_signature(self):
    #     # Create a dummy input for the model based on expected input shape
    #     input_text = "hi how are you"
    #     input_data = self.vectorizer.transform([input_text])
    #     input_df = pd.DataFrame(input_data.toarray(), columns=[str(i) for i in range(input_data.shape[1])])

    #     # Predict using the new model to verify the input and output shapes
    #     prediction = self.new_model.predict(input_df)

    #     # Verify the input shape
    #     self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))

    #     # Verify the output shape (assuming binary classification with a single output)
    #     self.assertEqual(len(prediction), input_df.shape[0])
    #     self.assertEqual(len(prediction.shape), 1)  # Assuming a single output column for binary classification

    def test_model_performance(self):
        # Extract features and labels from holdout test data
        X_holdout = self.holdout_x_data
        y_holdout = self.holdout_y_data

        # Predict using the new model
        y_pred_new = self.new_model.predict(X_holdout)

        # Calculate performance metrics for the new model
        r2_score_new = r2_score(y_holdout, y_pred_new)
        mae_new = mean_absolute_error(y_holdout, y_pred_new)


        # Define expected thresholds for the performance metrics
        expected_r2_score= 0.75
        expected_mae = 0.30

        # Assert that the new model meets the performance thresholds
        self.assertGreaterEqual(r2_score_new, expected_r2_score , f'Accuracy should be at least {expected_r2_score}')
        self.assertLessEqual(mae_new, expected_mae, f'Precision should be at least {expected_mae}')

if __name__ == "__main__":
    unittest.main()