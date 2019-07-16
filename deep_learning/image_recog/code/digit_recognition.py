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

train = pd.read_csv('../data/digit_train.csv')
test = pd.read_csv('../data/digit_test.csv')

y_train = train["label"]
print(y_train)
x_train = train.drop(labels=["label"], axis=1)
g = sns.countplot(y_train)
plt.show()
print(y_train.value_counts())