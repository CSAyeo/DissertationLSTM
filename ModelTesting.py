import pandas as pd
from sklearn.preprocessing import StandardScaler

import Model

data = pd.read_csv('Data\Test.csv', index_col=0)
print(data)

for i in range(len(data.columns)):  # key error of 3
    nModel = Model.model(3, 1, 300, 32, 100, 0.8)
    print(f"{i=}")
    tdata = data.loc[:, ['Divison {}'.format(i)]]
    print(tdata)
    scale = StandardScaler()
    scale.fit(tdata)
    tdata = scale.transform(tdata)
    x_test, y_test, x_train, y_train = nModel.SplitData(tdata)
    nModel = nModel.train(x_train, y_train)
    predicted = nModel.predict(x_test)
    predicted = Model.model.InverseData(nModel,predicted)
    print(predicted)
