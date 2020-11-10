import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
import util as GA
import numpy as np

li = ['X', 'Y']
for i in range(1, 221):
    li.append(str(i))
x_data = pd.read_csv('DataSet/indianPines_X.csv', usecols=li)
y_data = pd.read_csv('DataSet/indianPines_Y.csv', usecols=['class'])
cr = np.logspace(-2, 10, 13)
gr = np.logspace(9, 3, 13)
p_grid = dict(gamma=gr, C=cr)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(),param_grid=p_grid, cv=cv)
grid.fit(x_data, y_data)
print(grid.best_params_)