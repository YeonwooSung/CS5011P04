from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
import pandas as pd
import sys


def train_classifier(file_path='./tickets.csv'):
    print("Creating classifier...")

    #Creating a new classifier
    clf = MLPClassifier(solver='sgd', learning_rate_init=0.5, hidden_layer_sizes=(5,), verbose=True, momentum=0.3, activation='logistic', n_iter_no_change=5000, max_iter=10000)
    print(clf)

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
    enc = enc.fit(df_countries)  # Encoding locations
    encoded_countries = enc.transform(df_countries).toarray()
    print("Data encoded.")

    print("\nTraining...")
    clf.fit(X.as_matrix(), encoded_countries) # Training classifier on encoded data

    print("\nTraining score: %f" % clf.score(X.as_matrix(), encoded_countries))  # Scoring classifier

    model_name = 'output'
    print("\nSaving model as " + model_name + ".joblib")  # Saving model
    joblib.dump(clf, open('./' + model_name + '.joblib', 'wb'))
    print("Model saved.")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
        print("Using file : " + file_path)
        train_classifier(file_path)
    else:
        print("Using file : tickets.csv")
        train_classifier()

