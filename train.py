import os
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from sklearn.model_selection import train_test_split

processed_path = 'processed'

images_path = os.path.join(processed_path, 'images.npy')
labels_path = os.path.join(processed_path, 'labels.npy')

images = np.load(images_path)
labels = np.load(labels_path)

model = Sequential([
    Conv2D(32, 3, activation='relu', input_shape=images.shape[1:]),
    MaxPooling2D(2,2),
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, 3, padding='same', activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.3), 
    Dense(256, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.1)

def train():
	history = model.fit(
		train_images,
		train_labels,
		epochs=50
	)
	model.save('models/main.h5')
	print("Model trained successfully")

def test():
	load_model('models/main.h5')
	metrics = model.evaluate(test_images, test_labels, verbose=0, return_dict=True)
	print('Accuracy: {accuracy}\nLoss: {loss}'.format(**metrics))

train()
test()