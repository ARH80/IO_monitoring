import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D

np.random.seed(123)

# Model Definition
model = Sequential()

model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training Process

n_epochs = 10
n_duplication = 5

for i in range(n_epochs):

    X_train, y_train = None, None

    for _ in range(n_duplication):
        (X_train_new, y_train_test), (X_test, y_test) = mnist.load_data()

        if X_train is None:
            X_train, y_train = X_train_new, y_train_test
        else:
            y_train = np.concatenate((y_train, y_train_test))
            X_train = np.concatenate((X_train, X_train_new))

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    Y_train = to_categorical(y_train, 10)
    Y_test = to_categorical(y_test, 10)

    model.fit(X_train, Y_train, batch_size=32, nb_epoch=1, verbose=1)
