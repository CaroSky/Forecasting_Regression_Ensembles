from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from dataset_handler import DatasetHandler


class RandomForestRegressorHandler:
    def __init__(self, dataset_handler, n_trees=100, random_state=42):
        self.dataset_handler = dataset_handler
        self.n_trees = n_trees
        self.random_state = random_state
        # Initialize multiple decision trees with different random states
        self.trees = [DecisionTreeRegressor(random_state=np.random.randint(10000)) for _ in range(n_trees)]
        # Split data into training and evaluation sets
        self.X_train, self.X_eval, self.y_train, self.y_eval = self.dataset_handler.split_data(test_size=0.2, random_state=random_state)

    def train(self):
        # Train each tree in the forest on a bootstrapped sample of the data
        for tree in self.trees:
            X_train, y_train = self.dataset_handler.bootstrap_sample() # Get a bootstrap sample
            tree.fit(X_train, y_train) # Fit the tree on the bootstrap sample

    def evaluate(self):
        # Evaluate the performance of the Random Forest model
        y_pred = [self.predict(x) for x in self.X_eval]
        mse = mean_squared_error(self.y_eval, y_pred)
        rmse = mean_squared_error(self.y_eval, y_pred, squared=False)
        mae = mean_absolute_error(self.y_eval, y_pred)
        r2 = r2_score(self.y_eval, y_pred)
        # Print evaluation metrics
        print(f"Evaluation values from Random Forest: ")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R-squared (R2) Score: {r2} \n")

    def predict(self, X):
        # Aggregate predictions from all trees in the forest
        predictions = [tree.predict([X]) for tree in self.trees]
        return np.mean(predictions) # Return the mean of all predictions

    def visualize_prediction(self):
        # Visualization of the Random Forest predictions
        date_dataset = self.dataset_handler.data["Date"]
        y_dataset = self.dataset_handler.data["Demand"]
        date = []
        y_true = []
        y_prediction_rf = []

        for i in range(len(y_dataset) - self.dataset_handler.window_size):
            current_window = y_dataset[i:i + self.dataset_handler.window_size].values.tolist()
            date.append(datetime.strptime(date_dataset[i], "%d.%m.%Y"))
            y_prediction_rf.append(self.predict(current_window))
            y_true.append(y_dataset[i + self.dataset_handler.window_size])

        plt.plot(date, y_true, label='True')
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

    # Initialize and train the Random Forest Regressor Handler
    rf_regressor_handler = RandomForestRegressorHandler(dataset_handler, n_trees=100, random_state=42)
    rf_regressor_handler.train()
    rf_regressor_handler.evaluate()

    # Visualize the prediction results
    rf_regressor_handler.visualize_prediction()
