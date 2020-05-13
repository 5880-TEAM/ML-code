import numpy as np
import pandas as pd

from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

class RNN():
    def __init__(self, train_X, train_y, test_X, test_y):
        # design network
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.model = Sequential()
        self.model.add(LSTM(50, dropout=0.25, recurrent_dropout=0.25,
                            input_shape=(self.train_X.shape[1],
                                         self.train_X.shape[2])))
        self.model.add(Dense(1))
        self.model.compile(loss='mse', optimizer='adam')
    def learn(self):
        # fit network
        history = self.model.fit(self.train_X, self.train_y, epochs=20,
                                 validation_data=(self.test_X, self.test_y),
                                 verbose=2)
        plt.plot(history.history['loss'], label='train error')
        plt.plot(history.history['val_loss'], label='validation error')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('RNN Training History')
        plt.legend()
        plt.show()
    def forecast(self):
        yhat = self.model.predict(self.test_X)
        return yhat

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    (Adapted from Jason Brownlee's LSTM for multivariate time series)

    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).

    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var{}(t-{})'.format(j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var{}(t)'.format(j+1)) for j in range(n_vars)]
        else:
            names += [('var{}(t+{})'.format(j+1, i)) for j in range(n_vars)]
    # put it all together
    supervised = pd.concat(cols, axis=1)
    supervised.columns = names
    # drop rows with NaN values
    if dropnan:
        supervised.dropna(inplace=True)
    return supervised

def predictionmodel(dataset):
    # get dataset values
    values = dataset.values.reshape(-1, 1)

    # ensure all data is float
    values = values.astype('float32')

    # normalize features
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    # specify the number of lag samples
    n_days = 5
    n_features = values.shape[1]

    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_days, 1)

    # split into train and test sets
    values = reframed.values
    n_train_samples = round(0.6 * values.shape[0])
    train = values[:n_train_samples, :]
    test = values[n_train_samples:, :]

    #split into input and outputs
    n_obs = n_days * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    test_X, test_y = test[:, :n_obs], test[:, -n_features]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_days, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_days, n_features))

    # build neural network
    model = RNN(train_X, train_y, test_X, test_y)

    # begin learning process
    model.learn()

    # make a prediction
    yhat = model.forecast()

    # de-normalize predictions
    test_X = test_X.reshape((test_X.shape[0], n_days * n_features))
    inv_yhat = np.concatenate((yhat, test_X[:, -7:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    # de-normalize actual results
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -7:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    # calculate percent error
    score = 100 * (np.exp(np.sqrt(mean_squared_error(inv_y, inv_yhat))) - 1)
    score = round(score, 2)
    print('Average error: {}%'.format(score))
