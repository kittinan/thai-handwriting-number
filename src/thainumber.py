from PIL import Image, ImageOps
from keras.preprocessing.image import img_to_array, load_img, array_to_img
import os
import glob
import numpy as np
import pickle
import math


def clean_data():
    directory = "../data/clean"

    try:
        os.stat(directory)
    except:
        os.mkdir(directory)

    dirtys = [
        '7c9108fe-b240-4632-a024-f1ee922962ec',
        '20_a2178975-acff-4afe-88b9-f6fee8694ceb',
        'de366cab-6532-42ed-9926-38351927019b',
        '76c2e443-c8d1-40b0-96a9-073548c9617b',
        '21_e95ad3b8-30cb-47ad-9f68-2a5bb7aeb5bb',
        'e95ad3b8-30cb-47ad-9f68-2a5bb7aeb5bb',
        'fb05cb2a-c27b-4476-8cff-74f5ddbc8224',
        '078c1b18-e672-466d-a30b-f49a81710be6',
        '67ce79dc-de9c-4956-ad7b-fabf7aa9c6fa',
        '729207eb-f3f7-46e2-986a-74f990296da4',
        '420994cc-5e99-42eb-84b6-2392486a33b6',
    ]

    os.system('cp -r ../data/raw/thai-handwriting-number.appspot.com/* ../data/clean/')
    os.system('mv ../data/clean/10 ../data/clean/0')

    for i in range(0, 10):
        for dirty in dirtys:
            path = directory + '/' + str(i) + '/' + dirty + '.png'
            os.remove(path)


def make_dataset(data_dir = "../data/clean/", size = 28):
    X = []
    Y = []
    for folder in os.listdir(data_dir):
        if os.path.isdir(data_dir + folder) == True:
            label = folder
            for file in glob.glob(data_dir + folder + "/*.png"):
                img = load_img(file, grayscale=True, target_size=(size, size))
                img = ImageOps.invert(img)
                x = img_to_array(img)

                X.append(x)
                Y.append(label)
    X = np.asarray(X)
    Y = np.asarray(Y)
    data = {"X": X, "Y": Y};
    pickle.dump(data, open("thainumber_{}.pkl".format(size), "wb"), protocol = 2)

def load_dataset(size = 28):
    data = pickle.load(open("thainumber_{}.pkl".format(size), "rb"))
    X = data['X']
    Y = data['Y']
    return X, Y

def img_cloud_dataset(size = 28):
    X, Y = load_dataset(size)
    x = 0
    y = 0
    new_im = Image.new('L', (size * 50, size * math.ceil(X.shape[0] / 50)))
    for i in range(0, X.shape[0]):
        if (i != 0 and i % 50 == 0):
            y += size
            x = 0

        im = array_to_img(X[i])
        new_im.paste(im, (x, y))
        x += size
    new_im.save("cloud_dataset_{}.png".format(size))