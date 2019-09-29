from IPython.display import Image, SVG
import matplotlib.pyplot as plt

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras import regularizers

def show_images(random_test_images, encoded_imgs, decoded_imgs):
    plt.figure(figsize=(18,4))
    for i, image_idx in enumerate(random_test_images):
        # plot original image
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(x_test[image_idx].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plot encode image
        ax = plt.subplot(3, num_images, num_images + i + 1)
        plt.imshow(encoded_imgs[image_idx].reshape(8,4))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plot reconstructed image
        ax = plt.subplot(3, num_images, 2*num_images + i + 1)
        plt.imshow(decoded_imgs[image_idx].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

# Loads data, ignore class labels
(x_train, _), (x_test, _) = mnist.load_data()

# Scale data to range between 0 and 1
max_value = float(x_train.max())
x_train = x_train.astype('float32') / max_value
x_test = x_test.astype('float32') / max_value

# work with images as vectors: reshape 3D arrays as matrices
print(np.prod(x_train.shape[1:]))
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

input_dim = x_train.shape[1] # 784
encoding_dim = 32

compression_factor = float(input_dim)/encoding_dim

## SIMPLE AUTOENCODER

autoencoder = Sequential()
autoencoder.add(Dense(encoding_dim, input_shape=(input_dim,), activation='relu'))
autoencoder.add(Dense(input_dim, activation='sigmoid'))
autoencoder.summary()

# extract the encoder model to examine an encoded image
input_img = Input(shape=(input_dim,))
encoder_layer = autoencoder.layers[0]
encoder = Model(input_img, encoder_layer(input_img))

encoder.summary()

autoencoder.compile(optimizer='adam', loss="binary_crossentropy")
# input = output because this is image reconstruction
autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

num_images = 10
np.random.seed(11)
random_test_images = np.random.randint(x_test.shape[0], size=num_images)

encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

show_images(random_test_images, encoded_imgs, decoded_imgs)

## DEEP AUTOENCODER

deep_autoencoder = Sequential()

# Encoder Layers
deep_autoencoder.add(Dense(4 * encoding_dim, input_shape=(input_dim,), activation='relu'))
deep_autoencoder.add(Dense(2 * encoding_dim, activation='relu'))
deep_autoencoder.add(Dense(encoding_dim, activation='relu'))

# Decoder layers
deep_autoencoder.add(Dense(2 * encoding_dim, activation='relu'))
deep_autoencoder.add(Dense(4 * encoding_dim, activation='relu'))
deep_autoencoder.add(Dense(input_dim, activation='sigmoid'))

input_img = Input(shape=(input_dim,))
encoder_layer1 = deep_autoencoder.layers[0]
encoder_layer2 = deep_autoencoder.layers[1]
encoder_layer3 = deep_autoencoder.layers[2]
encoder = Model(input_img, encoder_layer3(encoder_layer2(encoder_layer1(input_img))))

deep_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
deep_autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, validation_data=(x_test, x_test))

num_images = 10
np.random.seed(11)
random_test_images = np.random.randint(x_test.shape[0], size=num_images)

encoded_imgs = encoder.predict(x_test)
decoded_imgs = deep_autoencoder.predict(x_test)

show_images(random_test_images, encoded_imgs, decoded_imgs)

# CONVOLUTIONAL AUTOENCODER

# reshape the images to 28 x 28 x 1 for the convnets:
# 1 channel because black and white
# if we had RGB, it would be 3
x_train = x_train.reshape((len(x_train), 28, 28, 1))
x_test = x_test.reshape((len(x_test), 28, 28 , 1))

cnn_autoencoder = Sequential()

# Encoder Layers
cnn_autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=x_train.shape[1:]))
cnn_autoencoder.add(MaxPooling2D((2, 2), padding='same'))
cnn_autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
cnn_autoencoder.add(MaxPooling2D((2, 2), padding='same'))
cnn_autoencoder.add(Conv2D(8, (3, 3), strides=(2,2), activation='relu', padding='same'))

# Flatten encoding for visualisation
cnn_autoencoder.add(Flatten())
cnn_autoencoder.add(Reshape((4, 4, 8)))

# Decoder Layers
cnn_autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
cnn_autoencoder.add(UpSampling2D((2, 2)))
cnn_autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
cnn_autoencoder.add(UpSampling2D((2, 2)))
cnn_autoencoder.add(Conv2D(16, (3, 3), activation='relu'))
cnn_autoencoder.add(UpSampling2D((2, 2)))
cnn_autoencoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

encoder = Model(inputs=cnn_autoencoder.input, outputs=cnn_autoencoder.get_layer('flatten_1').output)
cnn_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
cnn_autoencoder.fit(x_train, x_train, epochs=100, batch_size=128, validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = cnn_autoencoder.predict(x_test)

show_images(random_test_images, encoded_imgs, decoded_imgs)
