from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from dataset_handler import DatasetHandler
from decision_tree_regressor import DecisionTreeRegressorHandler
from random_forest_regressor import RandomForestRegressorHandler

class MLPRegressorEnsembleHandler:
    def __init__(self, dataset_handler, n_estimators=10, random_state=42):
        self.dataset_handler = dataset_handler
        self.n_estimators = n_estimators
        self.random_state = random_state
        # Initialize multiple MLP regressors with different random states
        self.regressors = [MLPRegressor(max_iter=1000, random_state=np.random.randint(10000)) for _ in range(n_estimators)]
        self.X_train, self.X_eval, self.y_train, self.y_eval = self.dataset_handler.split_data(test_size=0.2, random_state=random_state)

    def train(self):
        # Train each MLP regressor in the ensemble on a bootstrapped sample of the data
        for regressor in self.regressors:
            X_train, y_train = self.dataset_handler.bootstrap_sample()
            regressor.fit(X_train, y_train) # Fit the regressor on the bootstrap sample

    def predict(self, X):
        # Aggregate predictions from all MLP regressors in the ensemble
        predictions = [regressor.predict([X]) for regressor in self.regressors]
        return np.mean(predictions) # Return the mean of all predictions

    def evaluate(self):
        # Evaluate the performance of the MLP Regressor Ensemble
        y_pred = [self.predict(x) for x in self.X_eval]
        mse = mean_squared_error(self.y_eval, y_pred)
        rmse = mean_squared_error(self.y_eval, y_pred, squared=False)
        mae = mean_absolute_error(self.y_eval, y_pred)
        r2 = r2_score(self.y_eval, y_pred)
        # Print evaluation metrics
        print(f"Evaluation values from MLP Regressor: ")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R-squared (R2) Score: {r2} \n")

    def visualize_prediction(self, dt_regressor_handler, rf_regressor_handler):
        date_dataset = self.dataset_handler.data["Date"]
        y_dataset = self.dataset_handler.data["Demand"]
        date = []
        y_true = []
        y_prediction_mlp = []
        y_prediction_dt = []
        y_prediction_rf = []

        for i in range(len(y_dataset) - self.dataset_handler.window_size):
            current_window = y_dataset[i:i + self.dataset_handler.window_size].values.tolist()
            date.append(datetime.strptime(date_dataset[i], "%d.%m.%Y"))
            y_true.append(y_dataset[i + self.dataset_handler.window_size])
            y_prediction_mlp.append(self.predict(current_window))
            y_prediction_dt.append(dt_regressor_handler.model.predict([current_window])[0])
            y_prediction_rf.append(rf_regressor_handler.predict(current_window))

        plt.plot(date, y_true, label='True')
        plt.plot(date, y_prediction_mlp, label='MLP Ensemble Prediction')
        plt.plot(date, y_prediction_dt, label='Decision Tree Prediction')
        plt.plot(date, y_prediction_rf, label='Random Forest Prediction')

        plt.xlabel('Date')
        plt.ylabel('Demand')
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%Y"))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
        plt.gcf().autofmt_xdate()
        plt.show()

if __name__ == "__main__":
    csv_file_path = "data_2022.csv"
    target_column = "Demand"
    window_size = 100
    dataset_handler = DatasetHandler(csv_file_path, target_column, window_size)

    # Initialize and train DecisionTreeRegressorHandler
    dt_regressor_handler = DecisionTreeRegressorHandler(dataset_handler, random_state=42)
    dt_regressor_handler.train()
    dt_regressor_handler.evaluate()

    # Initialize and train RandomForestRegressorHandler
    rf_regressor_handler = RandomForestRegressorHandler(dataset_handler, n_trees=100, random_state=42)
    rf_regressor_handler.train()
    rf_regressor_handler.evaluate()

    # Initialize and train MLPRegressorEnsembleHandler
    mlp_ensemble_handler = MLPRegressorEnsembleHandler(dataset_handler, n_estimators=10, random_state=42)
    mlp_ensemble_handler.train()
    mlp_ensemble_handler.evaluate()

    # Visualize the predictions from all models
    mlp_ensemble_handler.visualize_prediction(dt_regressor_handler, rf_regressor_handler)