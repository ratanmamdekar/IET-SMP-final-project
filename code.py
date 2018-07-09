import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import operator


# read training file and create a list of it
def loadtrainingset(trainingSet=[]):
    with open("training.csv", "r", newline='') as csvfile:
        lines = csv.reader(csvfile)
        data = list(lines)
        for x in range(len(data)):
            for y in range(11):
                data[x][y] = float(data[x][y])
            trainingSet.append(data[x])


# read testing file and create a list of it
def loadtestingset(testingSet=[]):
    with open("testing.csv", "r", newline='') as csvfile:
        lines = csv.reader(csvfile)
        data = list(lines)
        for x in range(len(data)):
            for y in range(11):
                data[x][y] = float(data[x][y])
            testingSet.append(data[x])


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


# function that returns k most similar neighbors
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)

    for x in range(int(len(trainingSet))):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


# function for getting the majority voted response
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    votes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return votes[0][0]


# function that returns the accuracy
def getAccuracy(tesingtSet, predictions):
    correct = 0
    for x in range(len(testingSet)):
        num = testingSet[x][-1]
        if num == predictions[x]:
            correct += 1

    return (correct/float(len(testingSet))) * 100.0

# function for plotting bar graph of Predicted vs Actual data
def plot():
    # store the count of predicted and actual
    a = np.zeros((2, 10), dtype=int)
    for x in range(len(testingSet)):
        num = int(testingSet[x][-1])
        a[1][num] += 1
        if num == predictions[x]:
            a[0][num] += 1

    # data to plot
    n_groups = 10
    pk = list(range(10))
    pre = a[0]
    act = a[1]

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, pre, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Predicted')

    rects2 = plt.bar(index + bar_width, act, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Actual')

    plt.xlabel('Poker Hand')
    plt.ylabel('Number of instances')
    plt.title('Implementation of knn algorithm')
    plt.xticks(index + bar_width / 2, pk)
    plt.legend()

    plt.tight_layout()
    plt.show()


trainingSet = []
testingSet = []
loadtrainingset(trainingSet)
loadtestingset(testingSet)

print('Training set: ', len(trainingSet))
print('Testing set: ', len(testingSet))

predictions = list()
k = 5

for x in range(len(testingSet)):
    neighbors = getNeighbors(trainingSet, testingSet[x], k)
    result = getResponse(neighbors)
    predictions.append(result)
    print(x+1, '-- predicted=', result, ', actual=', testingSet[x][-1])
accuracy = getAccuracy(testingSet, predictions)
print('Accuracy: ', accuracy, '%')
plot()



