import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Dropout, Flatten, Reshape, PReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import callbacks, constraints

from evaluation_helpers import plot_reconstruction
from evaluation_helpers import plot_lines_of_best_fit, plot_predictions_histogram, plot_error_distribution, metrics_report

from config import imw, imh, c, RS, evaluation_path

"""
API for cell_autoencoder and regression
"""

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
    """
    fits @model with @data
    WARNING: this can be lengthy on a non-GPU local computer
    """

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
    """
    returns loss score for model
    returns visualisations on model reconstruction performance
    assumes model has been trained on @data and is being validated on @test
    """
    score = model.evaluate(data, data)
    print("Loss: {}".format(score))

    plot_reconstruction(model, data, tag)


def evaluate_regression(y_true, y_pred, y_labels, labels=["Unstimulated", "OVA", "ConA"], tag=None):
    """
    @x_test: input data to do predictions on
    @y_true: truth values for x_data
    @y_labels: categorical labels for nicer visualisations (not used in training, just informational)
    @tag: filename to save figures to
    """

    plot_lines_of_best_fit(y_true, y_pred, y_labels, labels, tag)
    plot_predictions_histogram(y_true, y_pred, y_labels, labels, tag)
    metrics_report(y_true, y_pred, y_labels, labels, tag)
    plot_error_distribution(y_true, y_pred, tag)
