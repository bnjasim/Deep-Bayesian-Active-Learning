from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.regularizers import l2 #, activity_l2

nb_classes = 10
# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 4

model = Sequential()
model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                 padding='valid',
                 activation='relu', 
                 input_shape=input_shape))

model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu'))

model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Conv2D(nb_filters*2, (nb_conv, nb_conv), 
                 padding='valid',
                 activation='relu'))

model.add(Conv2D(nb_filters*2, (nb_conv, nb_conv), activation='relu'))

model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

c = 2.5
Weight_Decay = c / 20
model.add(Flatten())
model.add(Dense(128, kernel_regularizer=l2(Weight_Decay), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])