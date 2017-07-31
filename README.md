# thai-handwriting-number
Thai handwriting number dataset


### Help us to create dataset
https://kittinan.github.io/thai-handwriting-number/

### Online predict thai number on web browser
https://kittinan.github.io/thai-handwriting-number/predict.html (Powered by [Keras-js](https://github.com/transcranial/keras-js))

## Instruction

#### Cleaning Data
Remove image is not match a number

```
cd src
python clean_data.py
```

#### Prepare data and train
- Resize all image into black and white (0 and 1) 28 x 28 px
- use Keras [mnist_cnn.py](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py) model to train

```
python train.py
```

## Contributing
Feel free to contribute on this project, I will be happy to work with you.

## License
MIT
