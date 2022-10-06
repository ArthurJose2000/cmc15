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
  classification_value = {}

  # append real data value
  classification_value["voter"] = voter

  if (random_value < percentage_of_republicans):
    # make a guess that the voter is republican
    classification_value["guess"] = "republican"

    if (voter == "republican"):
      # classification is correct
      classification_value["is_answer_correct"] = True
    else:
      # classification is wrong
      classification_value["is_answer_correct"] = False
  else:
    # make a guess that the voter is democrat
    classification_value["guess"] = "democrat"

    if (voter == "democrat"):
      # classification is correct
      classification_value["is_answer_correct"] = True
    else:
      # classification is wrong
      classification_value["is_answer_correct"] = False

  classification_list.append(classification_value)

print(classification_list)