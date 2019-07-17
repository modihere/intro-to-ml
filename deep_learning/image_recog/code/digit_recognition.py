import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itertools
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

sns.set(style='white', context='notebook', palette='deep')

#step 1: load data

train = pd.read_csv('../data/digit_train.csv')
test = pd.read_csv('../data/digit_test.csv')

y_train = train["label"]
print(y_train)
x_train = train.drop(labels=["label"], axis=1)
g = sns.countplot(y_train)
plt.show()
print(y_train.value_counts())

#step2: check for null and missing values

print(x_train.isnull().any().describe())
print(test.isnull().any().describe())

#step3: perfrom grayscale normalization

x_train = x_train/255
test = test/255

#step4: reshape as 1D vector

x_train = x_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

#step5: label the numbers

y_train = to_categorical(y_train, num_classes=10)

#step6: split into training and validation data

random_seed = 2

x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.1, random_state=random_seed)

plt.imshow(x_train[3][:,:,0])

#step7: create the model

model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu',input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#step8: setting learning rate and epochs

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
epochs = 5
batch_size = 84

#step 8: data augmentation to prevent the problem of overfitting

#starting with without data augmentation

history = model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val,y_val), verbose=2)

#here the code for data augmentation using ImageDataGenrator will be added

