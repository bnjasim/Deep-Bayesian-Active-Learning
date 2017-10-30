# Code by https://github.com/Xfan1025/Fashion-MNIST
# import essentials
import numpy as np
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.examples.tutorials.mnist import input_data
from keras.optimizers import Adam
opt = Adam(decay=0.001)

data = input_data.read_data_sets('MNIST_Fashion', one_hot=True)
x_train = data.train.images
y_train = data.train.labels
x_test = data.test.images
y_test = data.test.labels

img_rows, img_cols = 28, 28
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# check data shapes
print('train_features shape: ', x_train.shape)
print('test_features shape: ', x_test.shape)
print('train_labels shape: ', y_train.shape)
print('test_labels shape: ', y_test.shape)


# hyperparameters
epochs = 30
batch_size = 256
# build model
n_classes = 10

model = Sequential()
# input layer
model.add(InputLayer(input_shape=(28, 28, 1)))
# normalization
model.add(BatchNormalization())
model.add(Conv2D(64, (5, 5), activation='relu',
          bias_initializer='RandomNormal', kernel_initializer='random_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.35))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))
# compile the model
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

training = model.fit(x_train, y_train,
                     validation_data=(x_test, y_test),
                     epochs=epochs,
                     batch_size=batch_size, 
                     verbose=1)


score, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', accuracy)






