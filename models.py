import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.python.keras.metrics import Accuracy, Recall, Precision
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils import to_categorical, plot_model


def load_data_for_NN():
    mnist_train = pd.read_csv("data\\mnist_train.csv")
    mnist_test = pd.read_csv("data\\mnist_test.csv")
    y_train = mnist_train.label.values
    y_test = mnist_test.label.values
    x_train = mnist_train.values[:, 1:]
    x_test = mnist_test.values[:, 1:]
    # Standardization
    x_train = np.array([(x - x.mean()) / x.std() for x in x_train])
    x_test = np.array([(x - x.mean()) / x.std() for x in x_test])
    x_train = x_train.astype('float32').reshape(-1, 28, 28, 1)
    x_test = x_test.astype('float32').reshape(-1, 28, 28, 1)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test


def load_images():
    mnist = pd.read_csv("data\\mnist_test.csv")
    y = mnist.label.values
    x = mnist.values[:, 1:].astype('float32').reshape(-1, 28, 28)
    return x, y


def model_DNN():
    model = Sequential()
    model.add(Flatten(input_shape=[28, 28, 1]))
    model.add(Dense(512))
    model.add(Dense(128))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model


def model_CNN():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=[28, 28, 1]))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['acc'])
    return model


def get_checkpoint_path(model_name):
    checkpoint_path = model_name + "_training/cp.ckpt"
    return checkpoint_path


def fit_model(model, model_name, x_train, y_train, batch_size, num_epoch):
    checkpoint_path = get_checkpoint_path(model_name)
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                  save_weights_only=True,
                                  verbose=1,
                                  monitor="loss")
    model_log = model.fit(x_train, y_train, batch_size=batch_size, validation_split=0.3, epochs=num_epoch,
                          verbose=1, callbacks=[cp_callback])
    return model_log, model


def validate(model, x_train, y_train, x_test, y_test):
    score = model.evaluate(x_train, y_train, verbose=0)
    print('Train loss:', score[0], end=", ")
    print('Train accuracy:', score[1])
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0], end=", ")
    print('Test accuracy:', score[1])
    print(model.summary())


def save_model_plot(model, model_name):
    plot_model(model, to_file="images\\" + model_name + '_architecture.png', show_shapes=True, show_layer_names=False)


def plot_metric(model_log, metric_name, model_name):
    fig = plt.figure()
    fig.set_size_inches(6, 4)
    plt.plot(model_log.history[metric_name])
    plt.plot(model_log.history['val_' + metric_name])
    plt.title('model' + model_name + " - " + metric_name)
    plt.ylabel(metric_name)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.savefig("images\\" + model_name + '_' + metric_name + '.png')


if __name__ == "__main__":
    batch_size = 128
    num_epoch = 3

    print("----------------TRAINING----------------")

    x_train, y_train, x_test, y_test = load_data_for_NN()
    for model_name, model in [("CNN", model_CNN()), ("DNN", model_DNN())]:
        model_log, model = fit_model(model, model_name, x_train, y_train, batch_size, num_epoch)
        validate(model, x_train, y_train, x_test, y_test)
        save_model_plot(model, model_name)
        # "----------------EVALUATION----------------")
        for metric_name in ['accuracy', 'loss']:
            plot_metric(model_log, metric_name, model_name)

    # print("----------------LOADING----------------")

    # model_name = "CNN"
    # checkpoint_path = get_checkpoint_path(model_name)
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    #
    # model = model_CNN()
    # latest = tf.train.latest_checkpoint(checkpoint_dir)
    # model.load_weights(latest)

    # print(x_test[0].shape)
    #     # print(model.predict(x_test[0].reshape(1, 28, 28, 1)))


