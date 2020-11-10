import scipy.io as sp
import numpy as np
import pandas as pd

array = sp.loadmat('DataSet/Indian_pines.mat')
array = array["indian_pines"]
array = list(array)
npArray = np.empty(shape=(145 * 145, 222), dtype=int)
count = 0
for i in range(145):
    for j in range(145):
        li = list([i, j]) + list(array[i][j])
        npArray[count] = np.array(li)
        count += 1

columns = ['X', 'Y']
for i in range(1, 221):
    columns.append(i)
df = pd.DataFrame(data=npArray, columns=columns)
df.to_csv('Dataset/indianPines_X.csv')

array = sp.loadmat('DataSet/Indian_pines_gt.mat')
array = array["indian_pines_gt"]
array = list(array)
npArray = np.empty(shape=(145 * 145), dtype=int)
count = 0
for i in range(145):
    for j in range(145):
        npArray[count] = np.array(array[i][j])
        count += 1

columns = ['class']
df = pd.DataFrame(data=npArray, columns=columns)
df.to_csv('Dataset/indianPines_Y.csv')
