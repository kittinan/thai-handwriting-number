from tensorflow.python.lib.io import file_io
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import argparse
import pickle
import sys
import random
import h5py
from sklearn.cross_validation import train_test_split

from datetime import datetime

epochs = 50

def train_model(train_file = "./data/thainumber.pkl", job_dir='./tmp/thainumber', **args):
    print("Running Train Model")

    # set the logging path for ML Engine logging to Storage bucket
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('Using logs_path located at {}'.format(logs_path))

    #http://techqa.info/programming/question/40133223/Pickled-scipy-sparse-matrix-as-input-data-
    if sys.version_info < (3,):
        data = pickle.load(file_io.FileIO(train_file, mode='r'))
    else:
        data = pickle.loads(file_io.read_file_to_string(train_file))

    random_state = random.randint(1, 1024)
    x_train, x_test, y_train, y_test = train_test_split(data['X'], data['Y'], train_size=0.7, random_state=random_state)

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    num_classes = 10
    input_shape = (28, 28, 1)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Model
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

    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])


    batch_size = 128

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[])

    scores = model.evaluate(x_train, y_train)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    scores = model.evaluate(x_test, y_test)
    print("\nTEST %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # Save the weight of model locally
    model.save_weights('model_weight.h5')

    # Save the weight of model to the Cloud Storage bucket's jobs directory
    with file_io.FileIO('model_weight.h5', mode='r') as input_f:
        with file_io.FileIO(job_dir + '/model_weight.h5', mode='w+') as output_f:
            output_f.write(input_f.read())



if __name__ == '__main__':
    # Parse the input arguments for common Cloud ML Engine options
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--train-file',
      help='Cloud Storage bucket or local path to training data')
    parser.add_argument(
      '--job-dir',
      help='Cloud storage bucket to export the model and store temp files')
    args = parser.parse_args()
    arguments = args.__dict__
    print(arguments)
    train_model(**arguments)