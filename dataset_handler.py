import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DatasetHandler:
    def __init__(self, csv_file_path, target_column, window_size):
        self.data = pd.read_csv(csv_file_path)
        self.target_column = target_column
        self.window_size = window_size
        self.features = [col for col in self.data.columns if col != target_column]
        self.X, self.y = self.prepare_data()

    def prepare_data(self):
        """
        Prepare the dataset by applying a sliding window to create sequences of data.

        Returns:
            X (list of DataFrames): Features for the dataset with sliding windows.
            y (list of Series): Target variable for the dataset with sliding windows.
        """
        X_list = []
        y_list = []

        dataset = self.data[self.target_column].values

        for i in range(len(dataset) - self.window_size):
            X_window = dataset[i:i + self.window_size]
            y_window = dataset[i + self.window_size]
            X_list.append(X_window)
            y_list.append(y_window)

        return X_list, y_list

    def split_data(self, test_size=0.2, random_state=None):
        """
        Split the dataset with sliding windows into training and evaluation sets.

        Args:
            test_size (float): The proportion of the dataset to include in the evaluation set.
            random_state (int or None): Seed for the random number generator.

        Returns:
            X_train (list of DataFrames): Features for the training set.
            X_eval (list of DataFrames): Features for the evaluation set.
            y_train (list of Series): Target variable for the training set.
            y_eval (list of Series): Target variable for the evaluation set.
        """
        X_train, X_eval, y_train, y_eval = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)
        return X_train, X_eval, y_train, y_eval

    def bootstrap_sample(self):
        """
        Generates a bootstrapped sample from the training dataset. Bootstrapping is a method used
        in ensemble learning where a sample is chosen with replacement, meaning the same data point
        can be chosen multiple times. This method is used to introduce randomness into the training
        of ensemble models, which can help in improving their robustness and reducing overfitting.

        Returns:
            X_bootstrapped (list of DataFrames): Bootstrapped feature dataset.
            y_bootstrapped (list of Series): Bootstrapped target dataset.
        """
        indices = np.random.choice(len(self.X), size=len(self.X), replace=True)
        X_bootstrapped = [self.X[i] for i in indices]
        y_bootstrapped = [self.y[i] for i in indices]
        return X_bootstrapped, y_bootstrapped


    def get_data(self):
        """
        Get the entire dataset with sliding windows.

        Returns:
            X (list of DataFrames): Features for the entire dataset with sliding windows.
            y (list of Series): Target variable for the entire dataset with sliding windows.
        """
        return self.X, self.y

    def get_features(self):
        """
        Get the feature columns.

        Returns:
            features (list): List of feature column names.
        """
        return self.features

    def get_target_column(self):
        """
        Get the name of the target column.

        Returns:
            target_column (str): Name of the target column.
        """
        return self.target_column

