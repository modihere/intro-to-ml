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

ip_file = 'training.csv'


def create_dataset(dataset, look_back=1):
    datax, datay = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        datax.append(a)
        datay.append(dataset[i + look_back, 0])
    return np.array(datax), np.array(datay)


# np.random.seed(5)

# read input file
dataframe = pd.read_csv(ip_file, header=None, index_col=None, delimiter=',')
print(dataframe.shape)

# pull out the close price column[5]
all_y = dataframe[4].values
print(all_y)
# exit()
dataset = all_y.reshape(-1, 1)
print(dataset.shape)

# apply min max scalar
scalar = MinMaxScaler(feature_range=(0, 1))
dataset = scalar.fit_transform(dataset)

# split into train and test data
train_sz = int(len(dataset) * 0.7)
test_sz = len(dataset) - train_sz
print("test sz", test_sz)
train, test = dataset[0:train_sz, :], dataset[train_sz:len(dataset), :]

# shape the data with lookback
look_back = 300
trainx, trainy = create_dataset(train, look_back)
testx, testy = create_dataset(test, look_back)
print(trainx.shape, testx.shape)

# reshape the data the way LSTM should get
trainx = np.reshape(trainx, (trainx.shape[0], 1, trainx.shape[1]))  # batch size, no of features, loop_back
testx = np.reshape(testx, (testx.shape[0], 1, testx.shape[1]))

# LSTM modelling
model = Sequential()
model.add(
    LSTM(30, input_shape=(1, look_back)))  # 30 lstm cells with 300 previous time steps and 1 feature (closing price)
model.add(Dropout(0.15))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(trainx, trainy, epochs=1000, batch_size=400, verbose=1)

# predictions
trainpredict = model.predict(trainx)
testpredict = model.predict(testx)
print(testpredict.shape)

# invert predictions
trainpredict = scalar.inverse_transform(trainpredict)
trainy = scalar.inverse_transform([trainy])
testpredict = scalar.inverse_transform(testpredict)
testy = scalar.inverse_transform([testy])

# scores
trainscore = math.sqrt(mean_squared_error(trainy[0], trainpredict[:, 0]))
print('Train score %.2f RMSE' % (trainscore))
testscore = math.sqrt(mean_squared_error(testy[0], testpredict[:, 0]))
print('Test score %.2f RMSE' % (testscore))

# plotting graph
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainpredict) + look_back, :] = trainpredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainpredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testpredict

# plot baseline and predictions
plt.plot(scalar.inverse_transform(dataset))
plt.plot(trainPredictPlot)

# showing results
print('testprices:')
testprices = scalar.inverse_transform(dataset[train_sz + look_back + 1:])
print(testprices.size)
print('testpredictions:')
print(testpredict.size)
df = pd.DataFrame(data={"prediction": np.around(list(testpredict.reshape(-1)), decimals=2),
                        "test_price": np.around(list(testprices.reshape(-1)), decimals=2)})
print(df)

plt.plot(testPredictPlot)
plt.show()
