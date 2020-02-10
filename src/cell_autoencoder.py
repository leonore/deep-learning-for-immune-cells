import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, PReLU
from tensorflow.keras.models import Model

from plot_helpers import reshape, show_image

"""
API for cell_autoencoder

* make_autoencoder()
@returns:
- decoder: model to evaluate reconstruction capabilities
- encoder: model with flattened deepest, middle layer to feed into clustering algorithms

* train(model, data, batch_size, epochs)
- fits @model with @data. also called through evaluate() but can be called on its own

* evaluate(model, data, test, batch_size, epochs)
- returns visualisations on model reconstruction performance
- assumes model has been trained on @data and is being validated on @test

"""

imw = 192
imh = 192
c = 3
RS = 2211


def make_autoencoder():
    input_img = Input(shape=(imw, imh, c))

    x = Conv2D(64, (3, 3), padding='same')(input_img)
    x = PReLU()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = PReLU()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = PReLU()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), padding='same', strides=2)(x)
    x = PReLU()(x)

    encoded = Flatten()(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = PReLU()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = PReLU()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = PReLU()(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(c, (3, 3), activation='sigmoid', padding='same')(x)

    decoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    decoder.compile(optimizer='adam', loss='binary_crossentropy')

    return decoder, encoder


def train(model, data, batch_size=32, epochs=10):
    # get before/after weights (make sure there is a change)
    untrained_weights = np.array(model.get_layer(index=1).get_weights()[0])

    loss = model.fit(data, data, epochs=epochs, batch_size=batch_size)

    trained_weights = np.array(model.get_layer(index=1).get_weights()[0])

    # plot the loss
    plt.figure()
    plt.plot(loss.history['loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("Evolution of loss per epoch")
    plt.grid(True)
    plt.show()

    weight_diff = trained_weights - untrained_weights
    if np.all(weight_diff) == 0:
        print("Training does not seem to have changed the weights. Something might have gone wrong.")
    else:
        print("Model was trained successfully.")


def evaluate(model, data, test):
    plt.rcParams.update({'axes.titlesize': 'medium'})
    train_nb = np.random.randint(0, len(data)-1)
    test_nb = np.random.randint(0, len(test)-1)

    # show the difference in reconstruction
    decoded_imgs = model.predict(data[train_nb:train_nb+1]) # test on images it trained on
    untrained_decoded = model.predict(test[test_nb:test_nb+1]) # test images

    s=10
    fig = plt.figure(figsize=(s,s))
    fig.add_subplot(1, 2, 1)
    show_image(reshape(data[train_nb:train_nb+1], w=imw, h=imh, c=c), "original training image")
    fig.add_subplot(1, 2, 2)
    show_image(reshape(decoded_imgs[0], w=imw, h=imh, c=c), "reconstructed training image")

    fig = plt.figure(figsize=(s,s))
    fig.add_subplot(1, 2, 1)
    show_image(reshape(test[test_nb:test_nb+1], w=imw, h=imh, c=c), "original test image")
    fig.add_subplot(1, 2, 2)
    show_image(reshape(untrained_decoded[0], w=imw, h=imh, c=c), "reconstructed test image")

    plt.show()
