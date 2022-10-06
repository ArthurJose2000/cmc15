from re import T
import pandas as pd
from random import random

percentage_of_republicans = 0.548
percentage_of_democrats = 0.452

df = pd.read_csv('republican_democrat.csv', sep=',')

voters = df.pop('Target')

num_republicans = 0

classification_list = []


for voter in voters:
  random_value = random()
  
  if (random_value > percentage_of_republicans):
    # append real value  and a boolean to indicate if its correct
    # make a guess that the voter is republican
    if (voter == "republican"):
      classification_list.append({ voter, True })
    else:
      classification_list.append({ voter, False })
  else:
    # make a guess that the voter is democrat
    if (voter == "democrat"):
      classification_list.append({ voter, True })
    else:
      classification_list.append({ voter, False })

print(classification_list)