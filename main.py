import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dataset_handler import DatasetHandler
from decision_tree_regressor import DecisionTreeRegressorHandler
from random_forest_regressor import RandomForestRegressorHandler
from mlp_regressor import MLPRegressorEnsembleHandler

def predict_future_values(handler, test_dataset_handler, start_date, end_date):
    """
    Predicts future values for a given period using a trained model.

    Args:
        handler: The trained model handler (e.g., RandomForestRegressorHandler).
        test_dataset_handler: Handler for the test dataset.
        start_date (datetime): The start date for making predictions.
        end_date (datetime): The end date for making predictions.

    Returns:
        prediction_dates (list): List of dates for which predictions were made.
        predictions (list): List of predicted values.
    """
    current_data = test_dataset_handler.data["Demand"].tolist()
    last_date_in_data = datetime.strptime(test_dataset_handler.data.iloc[-1]["Date"], "%d.%m.%Y")
    current_date = last_date_in_data

    predictions = []
    prediction_dates = []
    while current_date < end_date:
        # Use sliding window approach for prediction
        if len(current_data) > test_dataset_handler.window_size:
            window_data = current_data[-test_dataset_handler.window_size:]

        # Make predictions within the specified date range
        if current_date >= start_date:
            predicted_demand = handler.predict(window_data)
            predictions.append(predicted_demand)
            current_data.append(predicted_demand)

        if current_date >= start_date:
            prediction_dates.append(current_date)

        current_date += timedelta(days=1)

    return prediction_dates, predictions


def plot_combined_predictions(dates, rf_predictions, dt_predictions, mlp_predictions):
    """
    Plots combined predictions from different models.

    Args:
        dates (list): List of dates for which predictions are made.
        rf_predictions (list): Predictions from the Random Forest model.
        dt_predictions (list): Predictions from the Decision Tree model.
        mlp_predictions (list): Predictions from the MLP Ensemble model.
    """
    plt.figure(figsize=(12, 6))
    # Plot predictions from each model
    plt.plot(dates, rf_predictions, label='Random Forest Predictions')
    plt.plot(dates, dt_predictions, label='Decision Tree Predictions')
    plt.plot(dates, mlp_predictions, label='MLP Ensemble Predictions')

    # Configure plot settings
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.title('Model Predictions for the Remainder of 2023')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%Y"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_csv_file_path = "data_2022.csv"
    test_csv_file_path = "data_2023.csv"
    target_column = "Demand"
    window_size = 100

    # Load and prepare training data
    train_dataset_handler = DatasetHandler(train_csv_file_path, target_column, window_size)

    # Load and prepare test data
    test_dataset_handler = DatasetHandler(test_csv_file_path, target_column, window_size)

    # Train and predict with RandomForestRegressorHandler
    rf_model = RandomForestRegressorHandler(train_dataset_handler, n_trees=100, random_state=42)
    rf_model.train()
    rf_dates, rf_predictions = predict_future_values(rf_model, test_dataset_handler, datetime(2023, 8, 26), datetime(2023, 12, 31))

    # Train and predict with DecisionTreeRegressorHandler
    dt_model = DecisionTreeRegressorHandler(train_dataset_handler, random_state=42)
    dt_model.train()
    dt_dates, dt_predictions = predict_future_values(dt_model, test_dataset_handler, datetime(2023, 8, 26), datetime(2023, 12, 31))

    # Train and predict with MLPRegressorEnsembleHandler
    mlp_model = MLPRegressorEnsembleHandler(train_dataset_handler, n_estimators=10, random_state=42)
    mlp_model.train()
    mlp_dates, mlp_predictions = predict_future_values(mlp_model, test_dataset_handler, datetime(2023, 8, 26), datetime(2023, 12, 31))

    # Plot combined predictions
    plot_combined_predictions(rf_dates, rf_predictions, dt_predictions, mlp_predictions)
