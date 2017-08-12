from keras.models import Sequential, model_from_json, load_model
import numpy as np
import sys
import argparse
import thainumber

def predict(model_weight="model.hdf5", file=""):

    print("Use weight: {}".format(model_weight))

    input = thainumber.prepare_input(file)

    json_string = open('model.json', 'r').read()
    model = model_from_json(json_string)
    model.load_weights(model_weight)

    result = model.predict(np.array([input]))
    print(result)

    result = model.predict_proba(np.array([input]))
    print(result)

    result = model.predict_classes(np.array([input]))
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--file',
      help='file image to predict')
    parser.add_argument(
      '--model-weight',
      help='Weight Mode H5 File',
      default='model.hdf5')
    args = parser.parse_args()
    arguments = args.__dict__
    predict(**arguments)

