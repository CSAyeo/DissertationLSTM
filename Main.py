import pandas as pd
import DataSimulator
import DataVisualiser
import Model
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from tensorflow import keras


# Create Decision Method
def Decision(DecList):
    dec = int(-1)
    q = DecList[0]
    del DecList[0]
    while not 0 <= dec < len(DecList):
        pos = 0
        print(q)
        for x in DecList:
            print(pos, " : ", x)
            pos += 1
        dec = int(input())
    return (dec)


# Handle CLI for Data Handling & Simulation
def GetPortfolioName():
    PN = input("What portfolio would you like to assess?")
    return PN


# Attempt to load the data specified by the user, if not found begin simulating instead
def Load(Portfolio):
    try:
        data = pd.read_csv('Data\{}.csv'.format(Portfolio), index_col=0)
    except:
        print("No found from that portfolio, simulating...")
        data = GetSimulated(Portfolio)
    return data


def GetSimOrLoad(Portfolio):
    SimOrLoad = Decision(["Generate or Simulate?", "Simulate", "Load"])
    if SimOrLoad:
        data = Load(Portfolio)
    else:
        data = GetSimulated(Portfolio)
    return data


def GetSimulated(Portfolio):
    Divisions = GetDivisions()
    Periods = GetPeriodsToSim()
    data = Simulate(Portfolio, Divisions, Periods)
    return data


def GetDivisions():
    DN = input("How many divisions within the portfolio")
    return DN


def GetPeriodsToSim():
    PTS = input("Number of periods to simulate? (Recommended 75)")
    return PTS


def Simulate(Portfolio, Divisions, Periods):
    return DataSimulator.Datagen(Portfolio, Divisions, Periods)


# Handle CLI for Model Creation
def CreateModel():
    nModel = Model.model(1, 1, 300, 32, 100, 0.8)  # initalise a model using configured params
    return nModel


def SelectDivision(data, Division):
    data = data.loc[:, ['Divison {}'.format(Division)]]  # take the first division
    return data


def ModelHandler(model, data):
    x_test, y_test, x_train, y_train = model.SplitData(data)
    nModel = model.train(x_train, y_train)
    return nModel, x_test, y_test


def TestModel(nModel, test, actual):
    predicted = nModel.predict(test)
    return predicted, actual


def DrawModel(model, data, i):
    nModel, test, actual = ModelHandler(model, data)
    predicted, actual = TestModel(nModel, test, actual)
    predicted, actual = model.InverseData(predicted), model.InverseData(actual)
    if CheckAccuracy(model, predicted, actual):
        DrawModel(model, data, i)
    return predicted


def CheckAccuracy(model, predicted, actual):
    final = predicted[0:-5]
    test = predicted[-5:]
    print(f"{final=} {test=}")
    Accuracy = round(GetAccuracy(predicted, actual), 2)
    return (Accuracy > 3.5)


def AllDivision(model, data, years):
    results = []
    for i in range(len(data.columns)):  # key error of 3
        tdata = data.loc[:, ['Divison {}'.format(i)]]
        tdata = model.ScaleData(tdata)
        results.append(DrawModel(model, tdata, i))

    return results

def GetPredictYears():
    YearsToPredict = int(input("How many years to predict"))
    if (YearsToPredict > 261):
        GetPredictYears()
    else:
        return 1

def GetAccuracy(predicted, actual):
    result = pd.DataFrame(predicted)
    result.columns = ['predict']
    result['actual'] = actual
    result['sum'] = (result['predict'] - result['actual']).abs()  # get difference between
    print(result)
    result['sum'] = (result['sum'] / result['actual']) * 100  # get percentage of accuracy
    print("Relative sum", result['sum'])
    ModelAccuracy = (result['sum'].sum() / len(result['sum']))
    return ModelAccuracy

def AddPredict(data, years):
    AddYears = pd.date_range(start=data.index[-1], periods=int(years), freq='A')
    AddYears = AddYears.date
    for i in range(1, years):
        print("Adding: ", AddYears[i])
        data.loc[AddYears[i]] = data.iloc[-1]
    return data

def DisplayPie():
    print("Pie")

def DisplayLine():
    print("Line")

def Vers():
    print("Christian Scavetta's Portfolio Guardrail Perdiction NN Vers 0.78")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
Vers()
Portfolio = GetPortfolioName()
data = GetSimOrLoad(Portfolio)
YearsToPredict = GetPredictYears()
data = AddPredict(data, YearsToPredict)
model = CreateModel()
results = AllDivision(model, data, YearsToPredict)
print(results)
DataVisualiser.vis(results)
