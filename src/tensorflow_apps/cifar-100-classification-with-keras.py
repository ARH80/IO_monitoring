import numpy as np

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

batch_size = 500
img_width, img_height, img_num_channels = 32, 32, 3
no_classes = 100
no_epochs = 10

# Model Definition

input_shape = (img_width, img_height, img_num_channels)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

loss_function = sparse_categorical_crossentropy
optimizer = Adam()

model.compile(
    loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

# Training Process

input_test, target_test = None, None
input_train, target_train = None, None

n_epochs = 2
n_duplication = 4

for i in range(n_epochs):

    input_test, target_test = None, None
    input_train, target_train = None, None

    for j in range(n_duplication):

        print(F"loading data [{i}, {j}]")
        (input_train_new, target_train_new), (input_test, target_test) = cifar100.load_data()

        if input_train is None:
            input_train, target_train = input_train_new, target_train_new
        else:
            input_train = np.concatenate((input_train, input_train_new))
            target_train = np.concatenate((target_train, target_train_new))

    input_train = input_train.astype('float32')
    input_test = input_test.astype('float32')

    input_train = input_train / 255
    input_test = input_test / 255

    history = model.fit(
        input_train, target_train,
        batch_size=batch_size,
        epochs=1,
        verbose=1,
        validation_split=0.2)

score = model.evaluate(input_test, target_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
