import os

import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib.image import imsave

from models import model_CNN, model_DNN, get_checkpoint_path


def model_predict(image: np.ndarray) -> int:
    model_name = "CNN"
    checkpoint_path = get_checkpoint_path(model_name)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if model_name == "CNN":
        model = model_CNN()
    elif model_name == "DNN":
        model = model_DNN()
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
    prediction = model.predict(image.reshape((1, 28, 28, 1)), verbose=0)
    result = np.argmax(prediction[0, :])
    return result


def random_image(x) -> (np.ndarray, int):  # -> pokazuje, jaki typ zmiennej zwraca funkcja
    no_of_images = x.shape[0]
    random_image_no = np.random.choice(no_of_images)
    return x[random_image_no, :, :], int(random_image_no)


def get_image_from_file():
    file_name = [file for file in os.listdir('static')][0]
    im = np.array(Image.open('static/' + file_name).convert('L'))
    return im, file_name


def delete_images_from_folder(folder_name):
    for file in os.listdir(folder_name):
        if file.endswith('.png'):
            os.remove(folder_name + '/' + file)


def render_image(x):
    delete_images_from_folder('static')
    image, image_no = random_image(x)
    image_name = 'image' + str(image_no) + '.png'
    imsave('static/' + image_name, image, cmap='gray')
    return image_name
