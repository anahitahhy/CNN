from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

(train_images, train_labels) , (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((6000, 784))
test_images = test_images.reshape((10000, 784))

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

train_images = to_categorical(train_labels)
test_images = to_categorical(test_labels)

network = Sequential()
network.add(Dense(512, activation = 'relu', input_shape=(784,)))
network.add(Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
network.fit(train_images, train_labels, epochs = 5, batch_size=128)

network.evaluate(test_images,test_labels)
