from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from warnings import simplefilter


# use the simplefilter to filter the FutureWarning
simplefilter(action='ignore', category=FutureWarning)


def train_classifier_XGBClassifier(file_path='./tickets.csv', plot_feature_importance=True):
    """
    This function trains the RandomForestClassifier for the Help Desk System.
    After finish training, this function generates the output file 'output.joblib'.

    :param file_path: The file path of the input file. (default value = './tickets.csv')
    :param plot_feature_importance: Boolean value to check if the user wants to plot the importance of each feature
    """
    print("Creating classifier...")

    # Creating a new classifier - RandomForestClassifier
    clf = RandomForestClassifier(random_state=3, n_jobs=-1, warm_start=True)
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

    # check if the program should plot the feature importance
    if (plot_feature_importance):
        # get the importances values of features by using the feature_importances_ attribute of RandomForestClassifier
        ftr_importances_values = clf.feature_importances_
        ftr_importances = pd.Series(ftr_importances_values)

        # plot the importance of each feature by using matplotlib and seaborn
        plt.figure(figsize=(8,6))
        plt.title('Feature importance')
        sns.barplot(y=ftr_importances, x=ftr_importances.index)
        plt.show()


if __name__ == "__main__":
    print("Using file : tickets.csv")
    train_classifier_XGBClassifier()

