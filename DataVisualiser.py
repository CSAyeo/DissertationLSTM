import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def vis(res):
    plt.subplot(2, 1, 1)
    c = 0
    for x in res:
        plt.plot(range(len(x)),x, label='division {}'.format(c))
        plt.legend()
        c+=1
    predicted = [sublist[-1].tolist() for sublist in res]
    predicted = np.reshape(predicted, -1)
    percent = [(predicted[i]/sum(predicted))*100 for i in range(len(predicted))]
    label = [f'Division {i}' for i in range(len(res))]
    plt.subplot(2, 1, 2)
    plt.pie(predicted, shadow=True, labels=label, autopct='%1.0f%%')
    plt.show()


