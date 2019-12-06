import time
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Activation, Dropout

scalar = None
np.random.seed(5)

def input_function(filename):
    print("Reading the input data")
    dataframe = pd.read_csv(filename, header=None, index_col=None, delimiter=',')
    print(dataframe.shape)
    return dataframe


def apply_scalar(dataset, scalar):
    print("Applying the MinMax Scalar")
    dataset = scalar.fit_transform(dataset)
    return dataset


def pick_columns(dataframe, scalar):
    print("Picking the close price column for training the model")
    all_y = dataframe[4].values
    dataset = all_y.reshape(-1, 1)
    dataset = apply_scalar(dataset, scalar)
    return dataset


def split_dataset(dataset):
    print("Splitting the dataset")
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    print(train_size, test_size)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    return train, test


def create_lookback_dataset(data, look_back=1):
    datax, datay = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), 0]
        datax.append(a)
        datay.append(data[i + look_back, 0])
    return np.array(datax), np.array(datay)


def lstm_ready_dataset(dataset, look_back):
    print("Making dataset LSTM ready")
    train, test = split_dataset(dataset)
    trainx, trainy = create_lookback_dataset(train, look_back)
    testx, testy = create_lookback_dataset(test, look_back)
    trainx = np.reshape(trainx, (trainx.shape[0], 1, trainx.shape[1]))  # batch size, no of features, loop_back
    testx = np.reshape(testx, (testx.shape[0], 1, testx.shape[1]))
    print(trainx.shape, trainy.shape, testx.shape, testy.shape)
    return trainx, trainy, testx, testy


def lstm_model(trainx, trainy, look_back):
    print("Modelling using LSTM...")
    model = Sequential()
    model.add(LSTM(30, input_shape=(
        1, look_back)))  # 30 lstm cells with 300 previous time steps and 1 feature (closing price)
    model.add(Dropout(0.15))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainx, trainy, epochs=100, batch_size=400, verbose=1)
    return model


def predictions(model, trainx, trainy, testx, testy):
    print("Predicting output on train and test data...")
    train_predict = model.predict(trainx)
    test_predict = model.predict(testx)

    train_predict = scalar.inverse_transform(train_predict)
    trainy = scalar.inverse_transform([trainy])
    test_predict = scalar.inverse_transform(test_predict)
    testy = scalar.inverse_transform([testy])
    print(train_predict.shape, test_predict.shape)
    return train_predict, test_predict, trainy, testy


def scoring(train_predict, test_predict, trainy, testy):
    print("Scoring are as follows... ")
    trainscore = math.sqrt(mean_squared_error(trainy[0], train_predict[:, 0]))
    testscore = math.sqrt(mean_squared_error(testy[0], test_predict[:, 0]))
    print("Train score %.2f RMSE" % (trainscore))
    print("Test score %.2f RMSE" % (testscore))
    return trainscore, testscore


def show_results(dataset, testpredict):
    print("Showing results...")
    train_size = int(len(dataset) * 0.7)
    print('test prices size:')
    testprices = scalar.inverse_transform(dataset[train_size + look_back + 1:])
    print(testprices.size)
    print('test predictions size:')
    print(testpredict.size)
    df = pd.DataFrame(data={"prediction": np.around(list(testpredict.reshape(-1)), decimals=2),
                            "test_price": np.around(list(testprices.reshape(-1)), decimals=2)})
    print(df)


def show_plots(dataset, trainpredict, testpredict, look_back):
    print("Plotting the data...")
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainpredict) + look_back, :] = trainpredict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainpredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testpredict

    # plot baseline and predictions
    plt.plot(scalar.inverse_transform(dataset), label='Closing price data')
    plt.plot(trainPredictPlot, label='Predictions on training')
    plt.plot(testPredictPlot, label='Predictions on test data')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    look_back = 300
    ip_file = 'training.csv'
    scalar = MinMaxScaler(feature_range=(0, 1))
    dataframe = input_function(ip_file)
    dataset = pick_columns(dataframe, scalar)
    trainx, trainy, testx, testy = lstm_ready_dataset(dataset, look_back)
    model = lstm_model(trainx, trainy, look_back)
    train_predict, test_predict, trainy, testy = predictions(model, trainx, trainy, testx, testy)
    trainscore, testscore = scoring(train_predict, test_predict, trainy, testy)
    show_results(dataset, test_predict)
    show_plots(dataset, train_predict, test_predict, look_back)
