import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv

keras = tf.keras

model = keras.models.load_model('dogs_vs_cats.h5')

tester = cv.imread('images/Dog1.jpg')
picture = tester
def format_Image(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (160, 160))
    return image

tester=format_Image(tester)


input_image_array = np.array(tester)
input_image_array = input_image_array[np.newaxis, :, :, :]

def DogOrCat(prediction):
    plt.imshow(picture)

    if prediction <0:
        cv.imshow('CAT', picture)
        print('CAT')
    else:
        cv.imshow('DOG', picture)
        print('DOG')
    cv.waitKey(10000)

DogOrCat(model.predict(input_image_array))


