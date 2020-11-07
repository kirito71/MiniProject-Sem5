import scipy.io as sp
import numpy as np
import pandas as pd

array = sp.loadmat('DataSet/Indian_pines_gt.mat')
array = array["indian_pines_gt"]
# print(len(array[0][0]))
nparray = np.empty(shape=145 * 145, dtype=int)
count = 0
for i in range(145):
    for j in range(145):
        nparray[count] = array[i][j]
        count += 1

columns = ['Class']
index = [i for i in range(145*145)]
df = pd.DataFrame(data=nparray, index=index, columns=columns)
df.to_csv('Dataset/indianPines_Y.csv')
# print(nparray)
