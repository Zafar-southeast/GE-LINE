# This is a script that creates random data into the form required to execute the script (Node Node Weight)
import numpy as np
import pandas as pd

N = 1000                                                             # Define the number of nodes of the dataset

####################################################################################################################
x1 = np.random.randint(0, 100, size=(N, ))                           # Created random integer for not node 1
x2 = np.random.randint(0, 100, size=(N, ))                           # Created random integer for not node 2
x3 = np.round(np.random.uniform(low=0.0, high=1.0, size=(N, )), 4)   # Created random float [0-1] for the weight

data = {'x1': x1, 'x2': x2, 'x3': x3}                                # Create dict with all values
df = pd.DataFrame.from_dict(data)                                    # Save  dict into a dataframe
df.to_csv('C:\\Users\\zafar\\PycharmProjects\\GE-LINE\\data\\Example\\RandomNew.txt', sep='\t', index=False, header=False)   # Save into folder NOTE WE NEED TO DEFINE THE PATH
####################################################################################################################
x11 = np.random.randint(0, 100, size=(N, ))                           # Created random integer for not node 1
x21 = np.random.randint(0, 100, size=(N, ))                           # Created random integer for not node 2
x31 = np.round(np.random.uniform(low=0.0, high=1.0, size=(N, )), 4)   # Created random float [0-1] for the weight

data = {'x1': x11, 'x2': x21, 'x3': x31}                                # Create dict with all values
df = pd.DataFrame.from_dict(data)                                    # Save  dict into a dataframe
df.to_csv('C:\\Users\\zafar\\PycharmProjects\\GE-LINE\\data\\Example\\RandomNew.txt', sep='\t', index=False, header=False)   # Save into folder NOTE WE NEED TO DEFINE THE PATH
