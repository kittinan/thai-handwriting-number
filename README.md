# thai-handwriting-number
Thai handwriting number dataset

![thai-handwriteing-number](https://raw.githubusercontent.com/kittinan/thai-handwriting-number/master/docs/img/thai-handwriting-number.png)


### Help us to create dataset
https://kittinan.github.io/thai-handwriting-number/

### Online predict thai number on web browser
https://kittinan.github.io/thai-handwriting-number/predict.html (Powered by [Keras-js](https://github.com/transcranial/keras-js))

## Requirement
- [Keras](https://keras.io/)
- [Tensorflow](https://www.tensorflow.org/)

## Instruction

#### Cleaning & Create Dataset
Remove image is not match a number, please see [thainumber.py](https://github.com/kittinan/thai-handwriting-number/blob/master/src/thainumber.py)

```python
import thainumber
thainumber.clean_data()

#Create dataset
thainumber.make_dataset()

#Load dataset
X,Y = thainumber.load_dataset()
```

#### Train
- use Keras [mnist_cnn.py](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py) model to train

```
python train.py
```

#### Train on Google Cloud ML Engine

Please see the folder [src/cloud-ml-engine](https://github.com/kittinan/thai-handwriting-number/tree/master/src/cloud-ml-engine) and this [blog](https://kittinanx.blogspot.com/2017/08/train-model-on-google-ml-engine.html)

#### Predict

```
python predict.py --file IMG_FILE
```



## Contributing
Feel free to contribute on this project, I will be happy to work with you.

## Thank you
[Thailand Deep Learning Facebook Group](https://www.facebook.com/groups/988867541235062/)

## License
MIT
