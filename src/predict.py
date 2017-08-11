from keras.models import Sequential, model_from_json
import numpy as np
import sys
import thainumber

file = sys.argv[1]

input = thainumber.prepare_input(file)

json_string = open('model.json', 'r').read()
model = model_from_json(json_string)
model.load_weights('model.hdf5')

result = model.predict(np.array([input]))
print(result)

result = model.predict_proba(np.array([input]))
print(result)

result = model.predict_classes(np.array([input]))
print(result)
