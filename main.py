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
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.9, random_state=1, stratify=y_data)
print(x_train.shape, x_test.shape)

# GA-SVM

population = GA.initPopulation()

for i in range(40):  # for 40 generations
    print(i)
    fitnessScores = GA.fitnessALL(x_train, y_train, population)
    zipList = zip(fitnessScores, population)
    zipList = sorted(zipList, reverse=True)
    population = [ele for _, ele in zipList]
    newPopulation = []
    for j in range(5):  # Choosing 5 elites from each population
        newPopulation.append(population[0])
        del population[0]
    population = population[:46]  # only 46 will be selected in next Population  --> Rank Selection
    count = 0
    while count < 46:
        parent1 = population[count]
        parent2 = population[count + 1]
        count = count + 2
        if GA.r.randint(0, 100) < 95:  # 95% crossOver rate
            offSpring1, offSpring2 = GA.crossOver(parent1, parent2)
        else:
            offSpring1, offSpring2 = parent1, parent2

        # Mutation
        if offSpring1 is None or offSpring2 is None:
            print('none', count)
        if GA.r.randint(0, 100) < 20:  # 20% Mutation rate
            offSpring1 = GA.mutate(offSpring1)
        if GA.r.randint(0, 100) < 20:  # 20% Mutation rate
            offSpring2 = GA.mutate(offSpring2)

        newPopulation.append(offSpring1)
        newPopulation.append(offSpring2)

    population = newPopulation

# After Genetics
print('After Genetics')
bestGene = population[0]
selectedFeatures = []
featureMask = bestGene[0][16:46]
count = 0
for bit in featureMask:
    if bit == '1':
        selectedFeatures.append(str(bestGene[1][count]))
    count += 1
x_train = x_train[selectedFeatures]
x_test = x_test[selectedFeatures]
C, gamma = GA.toPhenotype(bestGene)
model = SVC(C=C, gamma=gamma, kernel='rbf')
model.fit(x_train, y_train.values.ravel())
dump(model, 'trainedModel.joblib')
yPredict = model.predict(x_test)
ac = accuracy_score(y_test, yPredict)
print('Overall Accuracy:', ac)
print('Runtime:', time.time()-startTime)
