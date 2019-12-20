from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
import joblib
import pandas as pd
import sys
from warnings import simplefilter


# use the simplefilter to filter the FutureWarning
simplefilter(action='ignore', category=FutureWarning)


def train_classifier(file_path='./tickets.csv', do_cv=False):
    """
    This function trains the classifier for the neural network for the Help Desk System.
    After finish training, this function generates the output file 'output.joblib'.

    :param file_path: The file path of the input file. (default value = './tickets.csv')
    :param do_cv: The boolean value to check if this function should do the cross validation
    """
    print("Creating classifier...")

    # Creating a new classifier - Use MLPClassifier (Multi-Layer Perceptron Classifier)
    clf = MLPClassifier(solver='sgd', learning_rate_init=0.5, hidden_layer_sizes=(5,), verbose=True, momentum=0.3, activation='logistic', n_iter_no_change=5000, max_iter=10000)
    print(clf)

    print("\nReading file.")

    full_file = pd.read_csv(file_path)
    num_of_columns = len(full_file.columns)

    X_cols = [i for i in range(0, num_of_columns - 1)]

    # Reading only tag columns
    X = pd.read_csv(file_path, usecols=X_cols)

    # Reading only ResponseTeam column
    df_teams = pd.read_csv(file_path, usecols=[num_of_columns - 1])

    print("\nEncoding data...")
    X = X.replace(regex={r'Yes': 1, r'No': 0})  # Encoding tag data

    enc = OneHotEncoder(handle_unknown='ignore') # Use the One-Hot encoder to encode the input data
    enc = enc.fit(df_teams)  # Encoding response team
    encoded_team = enc.transform(df_teams).toarray()
    print("Data encoded.")

    print("\nTraining...")
    clf.fit(X.values, encoded_team)  # Training classifier on encoded data

    print("\nTraining score: %f" % clf.score(X.values, encoded_team))  # Scoring classifier

    # Saving model
    print("\nSaving model as output.joblib")
    joblib.dump(clf, open('./output.joblib', 'wb'))
    print("Model saved.")


    # check if the program should perform the cross validation
    if do_cv:
        # perform the cross validation
        cross_validate(clf, X, encoded_team)


def cross_validate(clf, x, y, cv=3):
    print("Start Running the Cross Validation")
    print('cross_val_score :', cross_val_score(clf, x, y, cv=cv))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
        print("Using file : " + file_path)
        train_classifier(file_path, True)
    else:
        print("Using file : tickets.csv")
        train_classifier(do_cv=True)

