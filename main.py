import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

li = ['X', 'Y']
for i in range(1, 221):
    li.append(str(i))
x_data = pd.read_csv('DataSet/indianPines_X.csv', usecols=li)
y_data = pd.read_csv('DataSet/indianPines_Y.csv', usecols=['class'])
print(x_data.shape, y_data.shape)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.6, random_state=1, stratify=y_data)
print(x_train.shape, x_test.shape)

model = KNeighborsClassifier()
model.fit(x_train, y_train)

yhat = model.predict(x_test)

print(accuracy_score(y_test, yhat))


