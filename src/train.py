import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image, ImageOps
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import random

def load_dataset():
    data_dir = "../data/clean/"
    X = []
    Y = []
    for folder in os.listdir(data_dir):
        if os.path.isdir(data_dir + folder) == True:
            label = folder
            for file in glob.glob(data_dir + folder + "/*.png"):
                #print(file)
                img = load_img(file, grayscale=True, target_size=(28, 28))
                img = ImageOps.invert(img)
                img = img.convert(mode="1")
                x = img_to_array(img)
                x = x.reshape((28, 28, 1))

                X.append(x)
                Y.append(label)

    return np.asarray(X), np.asarray(Y)

random_state = random.randint(1, 1024)

X,Y = load_dataset()
train_X, test_X, train_y, test_y = train_test_split(X, Y, train_size=0.7, random_state=random_state)

num_classes = 10
input_shape = (28, 28, 1)

train_y = keras.utils.to_categorical(train_y, num_classes)
test_y = keras.utils.to_categorical(test_y, num_classes)

#Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

epochs = 150
batch_size = 128
tbCallBack = keras.callbacks.TensorBoard(log_dir='./thai', histogram_freq=0, write_graph=True, write_images=True)

model.load_weights('model.hdf5')

model.fit(train_X, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(test_X, test_y),
          callbacks=[tbCallBack])

scores = model.evaluate(train_X, train_y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

scores = model.evaluate(test_X, test_y)
print("\nTEST %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


model.save_weights('model.hdf5')
config = model.to_json()
open("model.json", "w").write(config)
