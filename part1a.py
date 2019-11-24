from warnings import simplefilter
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
import joblib
import pandas as pd
import numpy as np
import sys
from warnings import simplefilter


# use the simplefilter to filter the FutureWarning
simplefilter(action='ignore', category=FutureWarning)


def train_classifier_with_GridSearchCV():
    file_path = './tickets.csv'
    print("Creating classifier...")

    parameters = {
        'solver': ['sgd'], 'learning_rate_init': [0.45, 0.5, 0.55], 'hidden_layer_sizes': np.arange(1,10), 
        'verbose': [True], 'momentum': [0.3], 'activation': ['logistic'], 'n_iter_no_change': [5000], 'max_iter': [1000]
    }

    # Use GridSearchCV to find the best parameter (i.e. the best number of hidden layers, etc)
    clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1)

    print("\nReading file.")

    full_file = pd.read_csv(file_path)
    num_of_columns = len(full_file.columns)

    X_cols = [i for i in range(0, num_of_columns - 1)]

    # Reading only question columns
    X = pd.read_csv(file_path, usecols=X_cols)

    # Reading only location column
    df_countries = pd.read_csv(file_path, usecols=[num_of_columns - 1])

    print("\nEncoding data...")
    X = X.replace(regex={r'Yes': 1, r'No': 0})  # Encoding question data

    enc = OneHotEncoder(handle_unknown='ignore') # Use the One-Hot encoder to encode the input data
    enc = enc.fit(df_countries)  # Encoding response team
    encoded_team = enc.transform(df_countries).toarray()
    print("Data encoded.")

    print("\nTraining...")
    clf.fit(X.values, encoded_team)  # Training classifier on encoded data

    print("\nTraining score: %f" % clf.score(X.values, encoded_team))  # Scoring classifier
    print("\nBest parameters: ", clf.best_params_) # print out the best parameters


if __name__ == "__main__":
    train_classifier_with_GridSearchCV()

