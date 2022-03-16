from random import random
import numpy as np
import pandas as pd

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

def Datagen(Portfolio, Divisions,  Periods):
    DataList = {}
    YearRange = pd.date_range(start='1/1/2000', periods=int(Periods), freq='A')
    for i in range(int(Divisions)):
        Startlist = []
        Start = int(GetStart(i))
        Trend = float(GetTrend(i))
        print((0.995 + Trend), (1.005 + (Trend)))
        for Year in range(int(Periods)):
            Start = Start * np.random.uniform((0.995 + Trend), (1.005 + (Trend)))
            Start = round(Start)
            Startlist.append(Start)
        print(Startlist)
        DataList['Divison {}'.format(i)] =Startlist
    print(DataList)
    df = pd.DataFrame(DataList, index=YearRange)
    print(df)
    df.index.name = 'Date'
    SaveDataDecision(df, Portfolio)
    return df

def GetTrend(Divisions):
    Trend = int(input("Please set trend for {} (50 is flat, 25 is recommended decrease, 75 is recommended increase)".format(Divisions)))
    Trend = (Trend - 50)
    print(Trend)
    Trend /= 25000
    print(Trend)
    Trend *= 3
    print(Trend)
    return Trend

def GetStart(Division):
    Start = input("Please specify starting value for division {} (Recommended 500)".format(Division))
    return Start

def SaveDataDecision(df, Portfolio):
    save = Decision(["Save data?", "Yes", "No"])
    if not save:
        SaveData(df, Portfolio)
        print(df)

def SaveData(Data, i):
    print("Saving data...")
    Data.to_csv("Data\{}.csv".format(i))

