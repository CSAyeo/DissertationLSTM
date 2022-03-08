import pandas as pd
import DataSimulator
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


def Load(Portfolio):
    try:
        data = pd.read_csv('{}.csv'.format(Portfolio),index_col=0)
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
    print("Returned data", data)
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
    nModel = Model.model(3, 1, 300, 32, 100, 0.8)  # initalise a model using configured params
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

def AllDivision(model, data):
    print(data)
    print(range(len(data.columns)))
    for i in range(len(data.columns)): #key error of 3
        tdata = data.loc[:, ['Divison {}'.format(i)]]
        tdata = model.ScaleData(tdata)
        nModel,  test, actual = ModelHandler(model, tdata)
        predicted, actual =TestModel(nModel, test, actual)
        predicted, actual = model.InverseData(predicted), model.InverseData(actual)
        print(GetAccuracy(predicted, actual))

def GetAccuracy( predicted, actual):
        result = pd.DataFrame(predicted)
        result.columns = ['predict']
        result['actual'] = actual
        result['sum'] = (result['predict'] - result['actual']).abs()#get difference between
        print(result)
        result['sum'] = (result['sum'] / result['actual']) * 100 # get percentage of accuracy
        print("Relative sum", result['sum'])
        ModelAccuracy = (result['sum'].sum()/len(result['sum']))
        return ModelAccuracy

Portfolio = GetPortfolioName()
data = GetSimOrLoad(Portfolio)
model = CreateModel()
AllDivision(model, data)