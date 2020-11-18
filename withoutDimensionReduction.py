import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import dump

li = ['X', 'Y']
for i in range(1, 221):
    li.append(str(i))
x_data = pd.read_csv('DataSet/indianPines_X.csv', usecols=li)
y_data = pd.read_csv('DataSet/indianPines_Y.csv', usecols=['class'])
print(x_data.shape, y_data.shape)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.9, random_state=1, stratify=y_data)
model = SVC(C=1e+4, gamma=1e-9, random_state=1, kernel='rbf')
model.fit(x_train, y_train.values.ravel())
dump(model, 'trainedModelMinus.joblib')
yPredict = model.predict(x_test)
ac = accuracy_score(y_test, yPredict)
print('Overall Accuracy:', ac)
