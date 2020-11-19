import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import dump
import util as GA
import time

startTime = time.time()
li = ['X', 'Y']
for i in range(1, 221):
    li.append(str(i))
x_data = pd.read_csv('DataSet/indianPines_X.csv', usecols=li)
y_data = pd.read_csv('DataSet/indianPines_Y.csv', usecols=['class'])
print(x_data.shape, y_data.shape)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.9, stratify=y_data)
print(x_train.shape, x_test.shape)

# GA-SVM

population = GA.initPopulation()
bestFitness = 0

for i in range(50):  # for 50 generations
    print(bestFitness, i)
    fitnessScores = GA.fitnessALL(x_train, y_train, population)
    zipList = zip(fitnessScores, population)
    population = sorted(zipList, reverse=True)
    bestFitness = population[0][0] * 100
    newPopulation = []
    for j in range(5):  # Choosing 5 elites from each population
        newPopulation.append(population[0][1])
        del population[0]

    count = 0
    population = population[:20]    # Rank Selection
    while count < 46:
        parents = GA.r.sample(population, k=2)
        parent1 = parents[0][1]
        parent2 = parents[1][1]
        count = count + 2
        if GA.r.randint(0, 100) < 95:  # 95% crossOver rate
            offSpring1, offSpring2 = GA.crossOver(parent1, parent2)
        else:
            offSpring1, offSpring2 = parent1, parent2

        # Mutation
        if GA.r.randint(0, 100) < 25:  # 25% Mutation rate
            offSpring1 = GA.mutate(offSpring1)
        if GA.r.randint(0, 100) < 25:  # 25% Mutation rate
            offSpring2 = GA.mutate(offSpring2)

        newPopulation.append(offSpring1)
        newPopulation.append(offSpring2)

    population = newPopulation

# After Genetics
print('After Genetics')
bestGene = population[0]
selectedFeatures = []
featureMask = bestGene[0][16:66]
count = 0
for bit in featureMask:
    if bit == '1':
        selectedFeatures.append(str(bestGene[1][count]))
    count += 1
x_train = x_train[selectedFeatures]
x_test = x_test[selectedFeatures]
C, gamma = GA.toPhenotype(bestGene)
model = SVC(C=C, gamma=gamma, kernel='rbf', cache_size=2000, decision_function_shape='ovo')
model.fit(x_train, y_train.values.ravel())
dump(model, 'trainedModel.joblib')
yPredict = model.predict(x_test)
ac = accuracy_score(y_test, yPredict)
print('Overall Accuracy:', ac)
print('Runtime:', time.time() - startTime)
