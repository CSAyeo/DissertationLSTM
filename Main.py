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
#@param DecList List with question in position 0, followed by answers
#Out, decision of user
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
#out; Portfolio Name as string
def GetPortfolioName():
    PN = input("What portfolio would you like to assess?")
    return PN



# Attempt to load the data specified by the user, if not found begin simulating instead
#@param Portfolio, name of the portfolio to load
def Load(Portfolio):
    try:
        data = pd.read_csv('Data\{}.csv'.format(Portfolio), index_col=0)
    except:
        print("No found from that portfolio, simulating...")
        data = GetSimulated(Portfolio)
    return data

#Let the user decide whether to load or simulate the data
#@param Portfolio name
def GetSimOrLoad(Portfolio):
    SimOrLoad = Decision(["Generate or Simulate?", "Simulate", "Load"])
    if SimOrLoad:
        data = Load(Portfolio)
    else:
        data = GetSimulated(Portfolio)
    return data

#Call the data Simulate funciton
#@params Portfolio name
def GetSimulated(Portfolio):
    Divisions = GetDivisions()
    Periods = GetPeriodsToSim()
    data = DataSimulator.Datagen(Portfolio, Divisions, Periods)
    return data


def GetDivisions():
    DN = input("How many divisions within the portfolio")
    return DN


def GetPeriodsToSim():
    PTS = input("Number of periods to simulate? (Recommended 75)")
    return PTS




def GetModelName():
    MN = input("Use Model:")
    return MN

# Handle CLI for Model Creation
def InitModel():
    mn = GetModelName()
    nModel = Model.model(5, 1, 300, 32, 100, 0.7, mn)  # initalise a model using configured params
    return nModel, mn


def SaveModel(model, mn):
    model.save('Model\{}.h5'.format(mn))


def SaveModelDec(model, mn):
    Save = Decision(["Save Model?", "Yes", "No"])
    if not Save:
        SaveModel(model, mn)


def SavePredictions():
    Save = Decision(["Save Predictions?", "Yes", "No"])
    if not Save:
        SaveModel()


def SelectDivision(data, Division):
    data = data.loc[:, ['Divison {}'.format(Division)]]  # take the first division
    return data


def ModelHandler(model, data):
    nModel,x_test, y_test  = model.SplitData(data)
    print(f"{y_test=}")
    return nModel, x_test, y_test



def DrawModel(model, data, i):
    nModel, xtest, ytest = ModelHandler(model, data)
    print(f"{ytest=} {xtest=}")
    predicted = nModel.predict(xtest)
    predicted, y_test = model.InverseData(predicted), model.InverseData(actual)
    if CheckAccuracy(model, predicted, y_test):
        DrawModel(model, data, i)
    return nModel, predicted


def CheckAccuracy(model, predicted, actual):
    final = predicted[0:-5]
    test = predicted[-5:]
    print(f"{final=} {test=}")
    Accuracy = round(GetAccuracy(predicted, actual), 2)
    return (Accuracy > 3.5)


def AllDivision(ModelObj,  data, years):
    results = []
    for i in range(len(data.columns)):  # key error of 3
        tdata = data.loc[:, ['Divison {}'.format(i)]]
        tdata = ModelObj.ScaleData(tdata)
        model, res = DrawModel(ModelObj, tdata, i)
        results.append(res)
    return model, results

def GetVisYears(max):
    YearsToVis= int(input("How many years to visualise"))
    if (YearsToVis > max):
        GetVisYears()
    else:
        return YearsToVis

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

def Vers():
    print("Christian Scavetta's Portfolio Guardrail Perdiction NN Vers 0.78")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
Vers()


ModelObj, mn = InitModel()
Portfolio = GetPortfolioName()
data = GetSimOrLoad(Portfolio)
YearsToVis = GetVisYears(len(data.index))
YearsToPredict=2
data = AddPredict(data, YearsToPredict)
model, results = AllDivision(ModelObj, data, YearsToPredict)
print(f"{results=} {data=}")
DataVisualiser.vis(results, YearsToVis)
SaveModelDec(model, mn)
print(f"{results=} {data=}")
DataSimulator.SaveDataDecision(data, Portfolio)