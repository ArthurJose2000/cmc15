from random import random
from constant import *

def get_percentages_votes(label_train):
  num_republicans = 0
  for voter in label_train:
    if voter == REPUBLICAN:
      num_republicans +=1
  
  percentage_of_republicans = num_republicans/len(label_train)
  return percentage_of_republicans, 1 - percentage_of_republicans

def classification(label_test, label_train):
  percentage_of_republicans, percentage_of_democrats = get_percentages_votes(label_train)

  voters = label_test

  test_list = []

  for voter in voters:
    random_value = random()
    classification_value = {}

    # append real data value
    classification_value["voter"] = voter

    if (random_value < percentage_of_republicans):
      # make a guess that the voter is republican
      classification_value["guess"] = REPUBLICAN

      if (voter == REPUBLICAN):
        # classification is correct
        classification_value["is_answer_correct"] = True
      else:
        # classification is wrong
        classification_value["is_answer_correct"] = False
    else:
      # make a guess that the voter is democrat
      classification_value["guess"] = DEMOCRAT

      if (voter == DEMOCRAT):
        # classification is correct
        classification_value["is_answer_correct"] = True
      else:
        # classification is wrong
        classification_value["is_answer_correct"] = False

    test_list.append(classification_value)

  print(test_list, len(test_list))