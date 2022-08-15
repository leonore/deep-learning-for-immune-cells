import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, PReLU
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks

from plot_helpers import reshape, show_image

"""
API for cell_autoencoder

* make_autoencoder()
@returns:
- decoder: model to evaluate reconstruction capabilities
- encoder: model with flattened deepest, middle layer to feed into clustering algorithms

* train(model, data, batch_size, epochs)
- fits @model with @data
- WARNING: this can be lengthy on a non-GPU local computer

* evaluate(model, data, test)
- returns visualisations on model reconstruction performance
- assumes model has been trained on @data and is being validated on @test

"""

from config import imw, imh, c, RS


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
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = PReLU()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), padding='same', strides=2)(x)
    x = PReLU()(x)

    encoded = Flatten()(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
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


def train(model, data, batch_size=32, epochs=20):
    # get before/after weights (make sure there is a change)
    untrained_weights = np.array(model.get_layer(index=1).get_weights()[1])

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.0001, verbose=1)
    early_stop = callbacks.EarlyStopping(monitor="val_loss", verbose=1, patience=5)

    loss = model.fit(data, data, epochs=epochs, batch_size=batch_size, validation_split=0.15,
                     callbacks=[reduce_lr, early_stop])

    trained_weights = np.array(model.get_layer(index=1).get_weights()[1])

    # plot the loss
    plt.figure()
    plt.plot(loss.history['loss'], label='loss')
    plt.plot(loss.history['val_loss'], label='val_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("Evolution of loss per epoch")
    plt.grid(True)
    plt.legend()
    plt.show()

    weight_diff = trained_weights - untrained_weights
    if np.all(weight_diff) == 0:
        print("Training does not seem to have changed the weights. Something might have gone wrong.")
    else:
        print("Model was trained successfully.")

def evaluate(model, data, tag=None):
    plt.rcParams.update({'axes.titlesize': 'medium'})
    test_nb = np.random.randint(0, len(test)-1)

    # show the difference in reconstruction
    decoded_imgs = model.predict(data[test_nb:test_nb+1])

    s=10

    fig = plt.figure(figsize=(s,s))
    fig.add_subplot(1, 2, 1)
    show_image(reshape(test[test_nb:test_nb+1], w=imw, h=imh, c=c), "original image")
    fig.add_subplot(1, 2, 2)
    show_image(reshape(untrained_decoded[0], w=imw, h=imh, c=c), "reconstructed image")

    plt.show()

    if tag:
        plt.save("../data/evaluation/autoencoder/" + tag + ".png")
