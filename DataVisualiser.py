import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def vis(res, disp):
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    c = 0
    for x in res:
        axs[0].plot(range(len(x)),x, label='division {}'.format(c))
        axs[0].legend()
        c+=1
    axs[0].set_title('')
    axs[0].set_xlabel('Time-Period')
    axs[0].set_ylabel('Value')
    predicted = [sublist[-1].tolist() for sublist in res]
    predicted = np.reshape(predicted, -1)
    label = [f'Division {i}: {round(predicted[i])}' for i in range(len(res))]
    print(f"{label=} {predicted=}")
    axs[1].pie(predicted, shadow=True, labels=label, autopct='%1.0f%%')
    axs[1].set_title('Predicted Division of Portfolio')
    plt.show()


