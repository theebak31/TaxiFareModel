from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from sklearn.linear_model import LinearRegression
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

        self.uri = "https://mlflow.lewagon.ai/"
        self.experiment_name = "[UK] [London] [theebak31] TaxiFareModel + v1.2"

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.uri)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


    def set_pipeline(self):
        print("defines the pipeline as a class attribute")
        dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())])

        time_pipe = Pipeline([
        ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))])

        preproc_pipe = ColumnTransformer([
        ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
        ('time', time_pipe, ['pickup_datetime'])])

        self.pipeline = Pipeline([
        ('preproc', preproc_pipe),
        ('linear_model', LinearRegression())])

    def run(self):
        print("set and train the pipeline")
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        print("evaluates the pipeline on df_test and return the RMSE")
        y_pred = self.pipeline.predict(X_test)

        rmse = np.sqrt(((y_pred - y_test)**2).mean())
        self.mlflow_log_metric("RMSE", rmse)

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        pass


if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate
    print('TODO')
