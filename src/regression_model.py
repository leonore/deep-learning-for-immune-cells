import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, PReLU
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import callbacks, constraints

"""
API for regression_model

* make_regression()
@returns:
- model: model that will attempt to predict an overlap value for the given image

* train(model, data, batch_size, epochs)
- fits @model with @data
- this regression model DEPENDS on the @encoder made in cell_autoencoder
- this will have to have been instantiated

* evaluate(model, x_test, y_true, y_labels, label=None)
args:
@model: regression model previously trained
@x_test: input data to do predictions on
@y_true: truth values for x_test
@y_labels: categorical labels for nicer visualisations (not used in training, just informational)
@label: filename to save figures to

@returns:
- visualisations and (TODO) TEXT on model regression performance

"""

def make_regression(encoder):
    model = Sequential()
    model.add(encoder)
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear', kernel_constraint=constraints.NonNeg()))

    model.compile(loss='mean_squared_error',
                  optimizer='adam')
    return model


def train(model, x_data, y_data, batch_size=64, epochs=20):
    # get before/after weights (make sure there is a change)
    untrained_weights = np.array(model.get_layer(index=1).get_weights()[1])

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.0001, verbose=1)
    early_stop = callbacks.EarlyStopping(monitor="val_loss", verbose=1, patience=5)

    loss = model.fit(x_data, y_data, epochs=epochs, batch_size=batch_size, validation_split=0.15,
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

def evaluate(y_true, y_pred, y_labels, tag=None):
    plot_lines_of_best_fit(y_true, y_pred, y_labels, tag)
    plot_predictions_histogram(y_true, y_pred, y_labels, tag)
    plot_error_distribution(y_true, y_pred, tag)
    metrics_report(y_true, y_pred, tag)
