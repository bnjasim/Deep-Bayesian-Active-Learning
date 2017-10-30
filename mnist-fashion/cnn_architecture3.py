# Code by: https://github.com/abelusha/MNIST-Fashion-CNN
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

batch_size     = 32
nb_classes     = 10
nb_ephochs     = 50
img_rows, img_cols = 28,28
nb_filters     = 32
pool_size      = 2
kernel_size    = 3
input_shape    = (img_rows, img_cols, 1)


model = Sequential()
model.add(Convolution2D(filters= nb_filters,kernel_size=(kernel_size,kernel_size),input_shape = input_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(pool_size,pool_size)))

model.add(Convolution2D(filters=nb_filters, kernel_size=(kernel_size,kernel_size),activation='relu'))
model.add(MaxPooling2D(pool_size=(pool_size,pool_size)))

model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(output_dim = 128 , activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(output_dim = nb_classes, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])