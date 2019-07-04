import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_images.shape)
print(train_labels,"\n",len(train_labels))
train_images = train_images / 255.0
test_images = test_images / 255.0
def plotting_heatmap():
	plt.figure()
	plt.imshow(train_images[0])
	plt.colorbar()
	plt.grid(False)
	plt.show()
def plotting_images():
	plt.figure(figsize=(10,10))
	for i in range(25):
		plt.subplot(5,5,i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(train_images[i], cmap=plt.cm.binary)
		plt.xlabel(class_names[train_labels[i]])
	plt.show()
def neural_model():
	model = keras.Sequential([
			keras.layers.Flatten(input_shape=(28,28)),
			keras.layers.Dense(128, activation=tf.nn.relu),
			keras.layers.Dense(128, activation=tf.nn.relu),
			keras.layers.Dense(10, activation=tf.nn.softmax)])
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
					metrics=['accuracy'])
	model.fit(train_images, train_labels, epochs=5)
	test_loss, test_acc = model.evaluate(test_images, test_labels)
	print ('Test accuracy :', test_acc)
	predictions = model.predict(test_images)
	print(predictions[1])
	print(np.argmax(predictions[1]))
	print(test_labels[1])
neural_model()
plotting_images()
