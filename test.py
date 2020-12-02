import pandas as pd
from sklearn.metrics import accuracy_score
import time
from joblib import load

startTime = time.time()
li = ['X', 'Y']
for i in range(1, 221):
    li.append(str(i))
x_data = pd.read_csv('DataSet/indianPines_X.csv', usecols=li)
y_data = pd.read_csv('DataSet/indianPines_Y.csv', usecols=['class'])
selected_Features = ['186', '144', '42', '58', '202', '76', '36', '32', '70', '176', '44', '48', '185', '14', '93', '68', '119', '26', '89', '102', '128', '25', '195', '99', '66', '181', '138', '164', '25', '95', '216', '193', '35', '55', '107', '134', '39', '132']
x_data = x_data[selected_Features]
model = load('trainedGaSvm.joblib')
yPredict = model.predict(x_data)
ac = accuracy_score(y_data, yPredict)
print('Overall Accuracy:', ac)
print('Runtime:', time.time() - startTime)