import sys
from warnings import simplefilter
from os import path

import part1
import part2
import part3


# use the simplefilter to filter the FutureWarning
simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 A4main.py <Bas|Int|Adv> [any param]")
    else:
        agentName = sys.argv[1]

        if (agentName == 'Bas'):
            print("Executing Basic Agent!")
            print("Using file : tickets.csv")
            part1.train_classifier()
        elif (agentName == 'Int'):
            print("Executing Intermediate Agent!")

            while True:
                # check if the "newTickets.csv" file exists
                if path.exists('./newTickets.csv'):
                    print('Use newTickets.csv')
                    part2.ask_user(file_to_use='./newTickets.csv')
                else:
                    print('Use newTickets.csv')
                    part2.ask_user()
        elif (agentName == 'Adv'):
            print("Executing Advanced Agent!")
            print("Using file : tickets.csv")
            part3.train_classifier_RandomForestClassifier()
