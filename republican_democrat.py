import math
import numpy as np
import pandas as pd
from a_priori import a_priori_classification

class dtree:
    def __init__(self, attributes, labels):
        self.parent = None
        # self.nodeGainRatio = 0
        # self.nodeInformationGain = 0
        self.isLeaf = False
        self.majorityClass = None
        self.bestAttribute = None
        self.children = {0: None, 1: None, 2: None}
        self.buildTree(attributes, labels)

    def buildTree(self, attributes, labels):
        numInstances = len(labels)
        nodeInformation = numInstances * computeEntropy(labels)
        self.majorityClass = mostFrequentlyOccurringValue(labels)

        if (nodeInformation == 0):  
            self.isLeaf = True
            return

        # Finding the best attribute for this node
        bestAttribute = None
        bestInformationGain = -math.inf
        bestGainRatio = -math.inf
        for attribute in attributes:
            conditionalInfo = 0
            attributeEntropy = 0
            attributeCount = {0: 0, 1: 0, 2: 0}
            for Y in set(attributes[attribute]):
                # Getting indexes of all instances for which attribute X == Y
                ids = segregate(attributes[attribute].tolist(), Y)
                attributeCount[Y] += len(ids)
                conditionalInfo += attributeCount[Y] * computeEntropy([labels[i] for i in ids])
            attributeInformationGain =  nodeInformation - conditionalInfo
            gainRatio = attributeInformationGain / computeEntropy(attributeCount)
            if (gainRatio > bestGainRatio):
                bestInformationGain = attributeInformationGain
                bestGainRatio = gainRatio
                bestAttribute = attribute

        # If no attribute provides any gain
        if (bestGainRatio == 0):
            self.isLeaf = True
            return

        # Else split by the best attribute
        self.bestAttribute = bestAttribute
        # self.nodeGainRatio = bestGainRatio
        # self.nodeInformationGain = bestInformationGain
        for Y in set(attributes[bestAttribute]):
            ids = segregate(attributes[bestAttribute].tolist(), Y)
            # Call a new children node for this node (now their parent)
            self.children[Y] = dtree(attributes.iloc[ids], [labels[i] for i in ids])
            self.children[Y].parent = self
        return
    
    def evaluate(self, testAttributes):
        if (self.isLeaf):
            return self.majorityClass
        else:
            # print(self.bestAttribute)
            return self.children[testAttributes[self.bestAttribute]].evaluate(testAttributes)

# Get indexes of the elements of attributearray that is equal to value
def segregate(attributearray, value):
    outlist = []
    for i, attribute in enumerate(attributearray):
        if attribute == value:
            outlist += [i]
    return outlist

# Calculate Entropy
def computeEntropy(labels):
    entropy = 0
    for i in range(2):
        probability_i = len(segregate(labels, i)) / len(labels)
        if probability_i != 0:
            entropy -= probability_i * math.log2(probability_i)
    return entropy
  
# Find most frequent value of a array
def mostFrequentlyOccurringValue(labels):
    bestCount = -math.inf
    bestId = None
    for i in range(2):
        count_i = len(segregate(labels,i))
        if (count_i > bestCount):
            bestCount = count_i
            bestId = i
    return bestId

# Get 2 arrays, compare and plot the confusion matrix
def confusionMatrix(actual, predicted):
    republican_republican = 0
    republican_democrat = 0
    democrat_republican = 0
    democrat_democrat = 0

    for index, value in enumerate(actual):
        if value == predicted[index]:
            if value == 0:
                republican_republican += 1
            if value == 1:
                democrat_democrat += 1
        else:
            if value == 0:
                republican_democrat += 1
            if value == 1:
                democrat_republican += 1

    print(f'           republican    {republican_republican:2}          {republican_democrat:2}')
    print(f'True Label   democrat    {democrat_republican:2}          {democrat_democrat:2}')
    print('                      republican   democrat')
    print('                        Predicted Label')
    return

# Priori confusion matrix
def prioriConfusionMatrix(actual, priori):
    republican_republican = 0
    republican_democrat = 0
    democrat_republican = 0
    democrat_democrat = 0

    for index, value in enumerate(actual):
        if value == priori[index]['guess']:
            if value == 0:
                republican_republican += 1
            if value == 1:
                democrat_democrat += 1
        else:
            if value == 0:
                republican_democrat += 1
            if value == 1:
                democrat_republican += 1

    print(f'           republican    {republican_republican:2}          {republican_democrat:2}')
    print(f'True Label   democrat    {democrat_republican:2}          {democrat_democrat:2}')
    print('                      republican   democrat')
    print('                        Predicted Label')
    return

# Split the dataset into 80% train and 20% test
def splitDataset(df):
    mask = np.random.rand(len(df)) < 0.8
    return df[mask].reset_index(drop=True), df[~mask].reset_index(drop=True)

# Turning categorical variables into numeric
def preProcessDataset(df):
    for column in df:
        unique_values = sorted(list(set(df[column])), reverse=True)
        for i, value in enumerate(df[column]):
            df[column][i] = unique_values.index(value)

    return df

def compareMeanSquareError(mean_absolute_error_1, mean_absolute_error_2):
    return (mean_absolute_error_1**2) / (mean_absolute_error_2**2)

# Cohen's kappa coefficient
def kappaCalculation(actual, dtree, priori):
    a = 0
    b = 0
    c = 0
    d = 0
    
    for index, value in enumerate(dtree):
        if value == priori[index]['guess']:
            if value == actual[index]:
                a += 1
            else:
                d += 1
        elif value == actual[index]:
            b += 1
        else:
            c += 1

    total = a + b + c + d
    po = (a + d)/(total)

    pYes = (a + b)*(a + c)/(total)**2
    pNo = (c + d)*(b + d)/(total)**2

    pe = pYes + pNo

    k = (po - pe)/(1 - pe)
    print(f'Kappa coefficient: {k}')

# Priori accuracy
def calculatePrioriAcc(priori):
    sum = 0
    for index in range(len(priori)):
        if priori[index]['is_answer_correct']:
            sum += 1
    return sum/len(priori)


if __name__ == '__main__':
    absolute_error_decisionTree = 0
    absolute_error_a_priori = 0

    print('Started!')
    # To grant reproducibility
    np.random.seed(100)

    # Loading the dataset csv
    df = pd.read_csv('republican_democrat.csv', sep=',')
    # df = pd.DataFrame([['y', 'n', '?', 'republican'], ['y', 'y', 'y', 'republican'],
    #                    ['n', '?', 'n', 'democrat'], ['y', '?', '?', 'democrat'], ['y', '?', 'y', 'democrat']],
    #                    columns=['a', 'b', 'c', 'Target'])
    
    print('\n--- Before pre-processing data ----')
    print(df)
    unique_values = sorted(list(set(df['Target'])), reverse=True)

    # Pre-processing the dataset
    df = preProcessDataset(df)
    print('\n--- After pre-processing data ----')
    print(df)

    # Spliting the dataset into train and test data
    df_train, df_test = splitDataset(df)
   
    print('\n--- Train Data ----')
    print(df_train)
    print('\n--- Test Data ----')
    print(df_test)

    # Separate into attributes and target label
    label_train = df_train.pop('Target')
    label_test = df_test.pop('Target')

    absolute_error_a_priori, priori = a_priori_classification(label_test, label_train)


    # Main Algorithm (Train Decision Tree)
    print('\n--- Training... ----')
    decision_tree = dtree(df_train, label_train)

    # Evaluate the model (Test Decision Tree)
    print('\n--- Evaluation - Decision Tree ----')
    print('index | Correct | Predicted')
    predicted_target_array = []
    correct = 0
    for index in range(len(df_test)):
        # print(df_test.loc[index])
        predicted_target = decision_tree.evaluate(df_test.loc[index])
        predicted_target_array += [predicted_target]
        if predicted_target == label_test[index]:
            correct += 1
        print(f' {index:2}      {str(predicted_target == label_test[index]):5}    {unique_values[predicted_target]}')

    absolute_error_decisionTree = len(df_test) - correct
    
    # Metrics
    print('\n--- Metrics----')
    print(f'Decision tree accuracy: {correct/len(df_test)*100:.2f} %\n')
    print(f'Priori accuracy: {calculatePrioriAcc(priori)*100:.2f} %\n')

    print('Decision tree confusion Matrix:')
    confusionMatrix(label_test.tolist(), predicted_target_array)

    print('\n--- Evaluation - mean square error comparation ----')
    print(f'MSE of a priori classication is {(compareMeanSquareError(absolute_error_a_priori, absolute_error_decisionTree) - 1)*100:.2f} % bigger than MSE of decision tree classication\n')
    
    print('Priori confusion Matrix:')
    prioriConfusionMatrix(label_test.tolist(), priori)

    kappaCalculation(label_test.tolist(), predicted_target_array, priori)
