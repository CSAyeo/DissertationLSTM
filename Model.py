import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM


class model:
    def __init__(self, ls, on, hn, bs, ep, pc, mn):
        self.length_of_sequences = ls
        self.in_out_neurons = on
        self.hidden_neurons = hn
        self.batch_size = bs
        self.epochs = ep
        self.percentage = pc
        self.scaler = preprocessing.StandardScaler()
        self.modelName = mn
    # prepare data

    def create_model(self):
        #creates the model using set params, evaluated for efficiency
        Model = Sequential()
        Model.add(LSTM(self.hidden_neurons, batch_input_shape=(None, self.length_of_sequences, self.in_out_neurons),
                       return_sequences=False))
        Model.add(Dense(self.in_out_neurons))
        Model.add(Activation("linear"))
        Model.compile(loss="mape", optimizer="adam")
        return Model


    # Param test code
    # def create_model(InitalTrainingSet, X_Train, Y_Train, DropoutRate, NodeCount, LayerCount):
    #     Model = Sequential()
    #     # Adding the first LSTM layer and some Dropout regularisation
    #     Model.add(LSTM(units=50, return_sequences=True, input_shape=(X_Train.shape[1], 1)))
    #     Model.add(Dropout(0.2))
    #     print(NodeCount)
    #     for x in range(LayerCount):
    #         Model.add(LSTM(units=NodeCount, return_sequences=True))
    #         Model.add(Dropout(DropoutRate))
    #     Model.add(LSTM(units=NodeCount))
    #     Model.add(Dropout(DropoutRate))
    #     Model.add(Dense(units=1))
    #     # Compiling the RNN
    #     Model.compile(optimizer='adam', loss='mean_squared_error')

    def DatatoArray(self, data, n_prev):
        x, y = [], []
        for i in range(len(data) - n_prev):
            x.append(data[i:(i + n_prev)])
            y.append(data[i + n_prev])
        X = np.array(x)
        Y = np.array(y)
        return X, Y

    def SplitData(self, data):
        split_pos = int(len(data) * self.percentage)
        x_train, y_train = self.DatatoArray(data[0:split_pos], self.length_of_sequences)
        x_test, y_test = self.DatatoArray(data[split_pos:], self.length_of_sequences)
        return x_test, y_test, x_train, y_train

    def ScaleData(self, data):
        self.scaler.fit(data)  # fir the scaler
        data = self.scaler.transform(data)  # transform the data
        return data

    def GetName(self):
        return self.modelName

    def InverseData(self, data):
        data = self.scaler.inverse_transform(data)  # transform the data
        return data

    def train(self, x_train, y_train):
            model = self.LoadModel()
            model.fit(x_train, y_train, self.batch_size, self.epochs)
            model.save('Model\{}.h5'.format(self.modelName))
            return model

    def LoadModel(self):
        try:
            model = keras.models.load_model('Model\{}.h5'.format(self.modelName))
        except:
            print("Error loading model, generating...")
            model = self.create_model()
        return model

