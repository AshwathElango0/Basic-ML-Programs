import tensorflow as tf         #imports required
import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Dense

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()        #loading data

train_data = train_data.reshape((60000, 28, 28, 1)).astype('float32')       #converting to floats for normalization and operations
test_data = test_data.reshape((10000, 28, 28, 1)).astype('float32')

train_labels = keras.utils.to_categorical(train_labels)         #one-hot encoding
test_labels = keras.utils.to_categorical(test_labels)

train_data /= 255.0                                         #normalization
test_data /= 255.0

model = keras.models.Sequential([Conv2D(32, (3, 3), activation='relu', input_shape = (28, 28, 1)),      #building  convolutional neural network
                                 MaxPooling2D((2, 2)),
                                 Conv2D(64, (3, 3), activation='relu'),
                                 MaxPooling2D((2, 2)),
                                 Conv2D(64, (3, 3), activation='relu'),
                                 keras.layers.Flatten(),
                                 Dense(64, activation='relu'),
                                 Dense(10, activation='softmax')])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])      #compiling with chosen loss and optimizer
model.fit(train_data, train_labels, epochs=3, batch_size=64)                #training

test_loss, test_acc = model.evaluate(test_data, test_labels, 32, verbose=1)         #testing
print(test_acc)                             #test results

