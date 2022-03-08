from random import random
import numpy as np
import pandas as pd

import CLI


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
    if SaveDataDecision():
        SaveData(df, Portfolio)
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

def SaveDataDecision():
    save = CLI.Decision(["Save data?", "Yes", "No"])
    return save

def SaveData(Data, i):
    print("Saving data...")
    Data.to_csv("{}.csv".format(i))

