import sys
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
import joblib
from warnings import simplefilter


# use the simplefilter to filter the FutureWarning
simplefilter(action='ignore', category=FutureWarning)


def ask_user():
    file_to_use = "./tickets.csv"        # file path of input file
    output_file_path = './output.joblib' # file path of weight file
    df = pd.read_csv(file_to_use)
    clf = MLPClassifier()

    # Loading classifier trained on output file (the joblib file)
    clf = joblib.load(output_file_path)

    # use OneHotEncoder to encode data
    enc = OneHotEncoder(handle_unknown='ignore')

    column_count = len(df.columns)
    df_teams = pd.read_csv(file_to_use, usecols=[column_count-1])
    column_count_div2Int = column_count // 2

    # Encoding response teams for decoding predicition
    enc.fit_transform(df_teams).toarray()

    answers = []
    count = 0

    # use for loop to iterate columns in the data frame
    for column in df.columns:
        if column == "Response Team":
            continue

        print(column + "?")  # Asking questions

        answer = ""

        # use while loop to loop until the user inputs suitable answer
        while answer != "y" and answer != "Y" and answer != "n" and answer != "N":
            answer = input("Please answer Y for yes, N for no: ") # gets the user input
            if answer == "Y" or answer == "y":
                answers.append(1)
            elif answer == "N" or answer == "n":
                answers.append(0)

        # increase the count value
        count = count + 1

        # check if count is equal to (column_count / 2) to check if the agent could make an early guess
        if count == column_count_div2Int:
            make_guess(answers, enc, clf, column_count)

    make_guess(answers, enc, clf, column_count)

    #TODO if the program not finished, then that means that the prediction is failed
    update_data(answers, file_to_use, column_count)


def make_guess(answers, enc, clf, column_count):
    current_answers = answers.copy()
    diff = column_count - 1 - len(current_answers)
    earlyPrediction = (diff > 0) # boolean value to check if this method is called to make an early guess

    # If  not all questions answered, floods list with 0's
    if earlyPrediction:
        for i in range(diff):
            current_answers.append(0)

    to_predict = [current_answers]
    prediction = enc.inverse_transform(clf.predict(to_predict))[0][0]  # Makes prediction

    # check if the program could make a prediction
    if prediction != None:
        print("I think the response team that could help you is " + prediction)
        is_correct = input("Am I correct?\n\t[y/n] >>")

        # use while loop to loop until the user inputs suitable answer
        while is_correct != "y" and is_correct != "Y" and is_correct != "n" and is_correct != "N":
            # gets the user input
            is_correct = input("Please answer Y for yes, N for no: ")

        # check the user input
        if is_correct == "y":
            print("Cool! I will route you to the " + prediction + " team right now.")
            sys.exit(1)
        elif is_correct == "n":
            if earlyPrediction:
                print("Oh dear. Let's continue...")
            else:
                print("That's a shame.")
    else:
        print("\nI have no guess..\n")

        if earlyPrediction:
            print("Let's continue!\n")


def update_data(answers, file_to_use, column_count):
    question_str = "Please tell me the name of the Reponse team that could help you: "

    # Gets correct answer
    response_team = input(question_str)

    yes = ""
    while yes != "y" and yes != "Y":
        yes = input("Is " + response_team + " correct?\n\t[y/n] >>")
        if yes == "n" or yes == "N":
            response_team = ""
            while response_team != "":
                response_team = input(question_str)

    yes = ""
    new_feature = ""
    while yes != "n" and yes != "N":  # Gets the new defining feature of new response team
        yes = input("Is there a defining feature to this response team that we have not asked about?\n\t[y/n] >>")
        if yes == "Y" or yes == "y":
            new_feature = input("Please tell me this new feature!: ")
            yes = "n"

    decoded_answers = [] # Decodes users answers

    # iterate the values in the list of answers
    for answer in answers:
        if answer == 1:
            decoded_answers.append("Yes")
        elif answer == 0:
            decoded_answers.append("No")

    #TODO ??


if __name__ == "__main__":
    ask_user()
