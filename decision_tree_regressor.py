from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from dataset_handler import DatasetHandler

class DecisionTreeRegressorHandler:
    def __init__(self, dataset_handler, random_state=42):
        # Initialization of the Decision Tree Regressor Handler
        self.dataset_handler = dataset_handler
        self.random_state = random_state
        self.model = DecisionTreeRegressor(random_state=random_state)

    def train(self):
        X_train, _, y_train, _ = self.dataset_handler.split_data(test_size=0.2, random_state=self.random_state)
        self.model.fit(X_train, y_train) # Fit the model on the training set

    def predict(self, X):
        # Predict using the trained Decision Tree model
        return self.model.predict([X])[0]

    def evaluate(self):
        # Evaluate the performance of the Decision Tree model
        _, X_eval, _, y_eval = self.dataset_handler.split_data(test_size=0.2, random_state=self.random_state)
        y_pred = self.model.predict(X_eval)
        # Calculate various evaluation metrics
        mse = mean_squared_error(y_eval, y_pred)
        rmse = mean_squared_error(y_eval, y_pred, squared=False)
        mae = mean_absolute_error(y_eval, y_pred)
        r2 = r2_score(y_eval, y_pred)
        # Print evaluation metrics
        print(f"Evaluation values from Decision Tree Regressor: ")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R-squared (R2) Score: {r2} \n")

    def visualize_prediction(self):
        # Visualization of the Decision Tree predictions
        date_dataset = self.dataset_handler.data["Date"]
        y_dataset = self.dataset_handler.data["Demand"]
        # Initialize lists for dates and predictions
        date = []
        y_true = []
        y_prediction_dt = []
        # Predict for each data point using sliding window
        for i in range(len(y_dataset) - self.dataset_handler.window_size):
            current_window = y_dataset[i:i + self.dataset_handler.window_size].values.tolist()
            date.append(datetime.strptime(date_dataset[i], "%d.%m.%Y"))
            y_prediction_dt.append(self.model.predict([current_window])[0])
            y_true.append(y_dataset[i + self.dataset_handler.window_size])
        # Plotting the actual vs predicted values
        plt.plot(date, y_true, label='True')
        plt.plot(date, y_prediction_dt, label='Decision Tree Prediction')
        # Configure plot settings
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

    dt_regressor_handler = DecisionTreeRegressorHandler(dataset_handler, random_state=42)
    dt_regressor_handler.train()
    dt_regressor_handler.evaluate()
    dt_regressor_handler.visualize_prediction()
