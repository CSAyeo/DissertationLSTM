import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def Datagen(years, type, start):
    Start = start
    Startlist = []
    Vals = [[0.995, 1.005], [0.993, 1.003], [0.997, 1.007]]
    for Year in range(years):
        Start = Start * np.random.uniform(Vals[type][0], Vals[type][1])
        print(Start)
        Start = round(Start)
        Startlist.append(Start)
    print(f"{Startlist=}")
    YearRange = pd.date_range(start='1/1/2000', periods=years, freq='A')
    df = pd.DataFrame(Startlist, index=YearRange, columns=['Guardrail'])
    return df


for i in range(3):
    df = Datagen(100, i, 500)
    df.plot()
    plt.show()