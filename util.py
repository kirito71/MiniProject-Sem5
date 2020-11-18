import random as r
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def binary(number, length=8):
    bi = bin(number).replace('0b', '')
    append = length - len(bi)
    append = '0' * append
    return append + bi


def rand_binary(p):
    key1 = ""
    for i in range(p):
        temp = str(r.randint(0, 1))
        key1 += temp

    return key1


def invert(a):
    if a == '1':
        return '0'
    else:
        return '1'


# pre defined C is gonna be 8 bits, gamma is 8 bits, featuresMask 30 bit(30 feature subset)
# 0 to 7 C, 8 to 15 gamma, 16 to 45 feature mask
def chromosome(C, gamma,
               features,
               featureMask):  # Each Chromosome will have 2 Dimensional data.. First dim represents gene and second dimension represent selected feature subset
    gene = binary(C)
    gene = gene + binary(gamma)
    gene = gene + featureMask
    return [gene, features]


def crossOver(gene1, gene2):
    pt = r.randint(1, 45)
    offspring1 = [gene1[0][:pt] + gene2[0][pt:46]]
    offspring2 = [gene2[0][:pt] + gene1[0][pt:46]]
    if pt <= 16:
        offspring1.append(gene2[1])
        offspring2.append(gene1[1])
    else:
        slicePt = pt - 16
        offspring1.append(gene1[1][:slicePt] + gene2[1][slicePt:30])
        offspring2.append(gene2[1][:slicePt] + gene1[1][slicePt:30])

    return offspring1, offspring2


def mutate(child, num_mut_bits=4):
    bits = r.sample(range(45), k=num_mut_bits)
    for bit in bits:
        temp = child[0]
        child[0] = temp[:bit]+invert(child[0][bit])+temp[(bit+1):]
    return child


def toPhenotype(gene, cMax=130, cMin=80, gammaMax=0.8, gammaMin=0.2):
    C = gene[0][:8]
    gamma = gene[0][8:16]
    C = int(C, 2)
    gamma = int(gamma, 2)
    C = cMin + ((cMax-cMin)/255)*C
    gamma = gammaMin + ((gammaMax - gammaMin) / 255) * gamma
    return C, gamma


def fitness(x_train, x_test, y_train, y_test, gene):
    selectedFeatures = []
    featureMask = gene[0][16:46]
    count = 0
    for bit in featureMask:
        if bit == '1':
            selectedFeatures.append(str(gene[1][count]))
        count += 1
    if len(selectedFeatures) == 0:
        return 0
    x_train, x_test = x_train[selectedFeatures], x_test[selectedFeatures]
    # print(x_train.shape, x_test.shape)
    C, gamma = toPhenotype(gene)
    model = SVC(C=C, gamma=gamma, kernel='rbf')
    model.fit(x_train, y_train.values.ravel())
    yPredict = model.predict(x_test)
    ac = accuracy_score(y_test, yPredict)
    # print(ac)
    return .7 * ac + .3 / len(selectedFeatures)


def initPopulation(length=100):
    population = []
    for i in range(length):
        C = r.randint(0, 255)
        gamma = r.randint(0, 255)
        featureMask = rand_binary(30)
        featureSubSet = r.sample(range(1, 221), k=30)
        gene = chromosome(C, gamma, featureSubSet, featureMask)
        population.append(gene)
    return population


def fitnessALL(x_data, y_data, population):
    score = []
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.7, stratify=y_data)
    for individual in population:
        score.append(fitness(x_train, x_test, y_train, y_test, individual))
    return score
