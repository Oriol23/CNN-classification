import os
#print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAIN_DIRECTORY = os.path.dirname(os.path.dirname(__file__))
DATA_DIRECTORY = os.path.join(MAIN_DIRECTORY,"data")

#print(DATA_DIRECTORY)