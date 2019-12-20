import sys
from warnings import simplefilter

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
                part2.ask_user()
                print("\n\nNow let's handle next ticket!")
        elif (agentName == 'Adv'):
            print("Executing Advanced Agent!")
            print("Using file : tickets.csv")
            part3.train_classifier_RandomForestClassifier()
