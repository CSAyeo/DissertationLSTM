import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def vis(res):
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    c = 0
    for x in res:
        axs[0].plot(range(len(x)),x, label='division {}'.format(c))
        axs[0].legend()
        c+=1
    axs[0].set_title('subplot 1')
    axs[0].set_xlabel('distance (m)')
    axs[0].set_ylabel('Damped oscillation')
    predicted = [sublist[-1].tolist() for sublist in res]
    predicted = np.reshape(predicted, -1)
    label = [f'Division {i}' for i in range(len(res))]
    axs[1].pie(predicted, shadow=True, labels=label, autopct='%1.0f%%')
    axs[1].set_title('subplot 2')
    plt.show()


