# Code by: https://github.com/Xfan1025/Fashion-MNIST
import keras
from keras.utils import to_categorical
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation, Flatten, Reshape, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import Adam
opt = Adam(decay=0.001)

batch_size     = 32
nb_classes     = 10
nb_ephochs     = 50
img_rows, img_cols = 28,28
nb_filters     = 32
pool_size      = 2
kernel_size    = 3
input_shape    = (img_rows, img_cols, 1)



model = Sequential()
# input layer
model.add(InputLayer(input_shape=(28, 28, 1)))
# normalization
model.add(BatchNormalization())
model.add(Conv2D(64, (5, 5), activation='relu',
          bias_initializer='RandomNormal', kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(nb_classes, activation='softmax'))
# compile the model
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
