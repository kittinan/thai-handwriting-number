import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image, ImageOps
import numpy as np
import sys

file = sys.argv[1]

def prep_input(file):
	img = load_img(file, grayscale=True, target_size=(28, 28))
	img = ImageOps.invert(img)
	img = img.convert(mode="1")
	x = img_to_array(img)
	x = x.reshape((28, 28, 1))
	return x

input = prep_input(file)

json_string = open('model.json', 'r').read()
model = model_from_json(json_string)
model.load_weights('model.hdf5')

result = model.predict(np.array([input]))
print(result)

result = model.predict_proba(np.array([input]))
print(result)

result = model.predict_classes(np.array([input]))
print(result)
