import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Dropout, Flatten, Reshape, PReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import callbacks, constraints

from evaluation_helpers import reshape, show_image
from evaluation_helpers import plot_lines_of_best_fit, plot_predictions_histogram
from evaluation_helpers import plot_error_distribution, metrics_report

"""
API for cell_autoencoder and regression

* make_autoencoder()
@returns:
- decoder: model to evaluate reconstruction capabilities
- encoder: model with flattened deepest, middle layer to feed into clustering algorithms

* make_regression()
@returns:
- model: model that will attempt to predict an overlap value for the given image

* train(model, data, batch_size, epochs)
- fits @model with @data
- WARNING: this can be lengthy on a non-GPU local computer

* evaluate_autoencoder(model, data, test)
- returns visualisations on model reconstruction performance
- assumes model has been trained on @data and is being validated on @test

* evaluate(model, x_test, y_true, y_labels, label=None)
args:
@model: regression model previously trained
@x_test: input data to do predictions on
@y_true: truth values for x_test
@y_labels: categorical labels for nicer visualisations (not used in training, just informational)
@label: filename to save figures to

@returns:
- visualisations and text on model regression performance

"""

from config import imw, imh, c, RS, evaluation_path


def make_autoencoder():
    """
    Initialise autoencoder model for training and
    return reference to both decoder and encoder parts of the model.
    """

    # image shape is defined in the configuration
    input_img = Input(shape=(imw, imh, c))

    # layers for reduction of image
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

    encoded = Flatten()(x) # bottleneck layer

    # layers for expansion of image
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = PReLU()(x)
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
    # the encoder will be trained through the decoder so it does not need to be compiled
    decoder.compile(optimizer='adam', loss='binary_crossentropy')

    return decoder, encoder

def make_regression(encoder):
    """
    Initialise a regression model for training using
    a previously created encoder model
    """

    model = Sequential()
    model.add(encoder)
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear', kernel_constraint=constraints.NonNeg()))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def train(model, x, y, batch_size=64, epochs=40, tag=None):
    # get before/after weights (make sure there is a change)
    untrained_weights = np.array(model.get_layer(index=1).get_weights()[1])

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.0001, verbose=1)
    early_stop = callbacks.EarlyStopping(monitor="val_loss", verbose=1, patience=5)

    loss = model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=0.15,
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

    if tag:
        plt.savefig(evaluation_path + tag + "_loss.png", dpi=300)

    plt.show()

    weight_diff = trained_weights - untrained_weights
    if np.all(weight_diff) == 0:
        print("Training does not seem to have changed the weights. Something might have gone wrong.")
    else:
        print("Model was trained successfully.")

def evaluate_autoencoder(model, data, tag=None):
    plt.rcParams.update({'axes.titlesize': 'medium'})
    test_nb = np.random.randint(0, len(data)-1)

    # show the difference in reconstruction
    decoded_imgs = model.predict(data[test_nb:test_nb+1])

    s=10

    fig = plt.figure(figsize=(s,s))
    fig.add_subplot(1, 2, 1)
    show_image(reshape(data[test_nb:test_nb+1], w=imw, h=imh, c=c), "original image [index {}]".format(test_nb))
    fig.add_subplot(1, 2, 2)
    show_image(reshape(decoded_imgs[0], w=imw, h=imh, c=c), "reconstructed image")

    if tag:
        plt.savefig(evaluation_path + "autoencoder/" + tag + "_reconstruction.png", dpi=300)

    plt.show()

def evaluate_regression(y_true, y_pred, y_labels, labels=["Unstimulated", "OVA", "ConA"], tag=None):
    plot_lines_of_best_fit(y_true, y_pred, y_labels, labels, tag)
    plot_predictions_histogram(y_true, y_pred, y_labels, labels, tag)
    plot_error_distribution(y_true, y_pred, tag)
    metrics_report(y_true, y_pred, tag)
