# Evaluating different autoencoder models

Please give:
* Structure (layers)
* Optimiser (incl. parameters if applicable)
* Loss function
* Epochs, other parameters
* Photos of reconstruction
* Does the loss value move?
* Do the weights of the layers change? (We want our encoder model to have learnt something)

## Model 1

### Structure

```python
x =  Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
encoded = Flatten()(x)

x =  Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x =  Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x =  Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), padding='same')(x)
```

### Parameters

* Optimiser = adam
* Loss function = binary_crossentropy
* n epochs = 30
* batch size = 48

### Images

![Image results](results/model1_output.png)
![Loss function](results/model1_loss.png)

### Results

* Loss does not seem to reach a plateau; need longer epoch
* Weights do not change; need more training data?

### Next up

* Augment dataset; see if it makes any changes with this basic model






---------------- old records --------------------

- add an extra convolutional layer, following each other: more lined, pixelly
- reduce filter for half the conv2d: more boxed
- more convolutional filters at beginning works better than less
- activation = sigmoid produces black
- smaller kernel size gets more pixellated results
- varying kernel size gets truer colour results
- bigger kernel size towards the centre loses points of where the bigger standouts are
- reducing maxpooling obviously keeps more detail

- the following works pretty well but doubles dimensionality:
```python
x =  Conv2D(128, (5, 5), activation='relu', padding='same')(input_img)
x =  Conv2D(128, (5, 5), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x =  Conv2D(128, (5, 5), activation='relu', padding='same')(x)
x =  Conv2D(128, (5, 5), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

decoded = Conv2D(1, (3, 3), padding='same')(x)
```

- this has a bit less detail but works pretty well. not sure about having two convolutions right after the other
```python
x =  Conv2D(128, (5, 5), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x =  Conv2D(128, (5, 5), activation='relu', padding='same')(x)
# this will help going back the original image dimensions
x = UpSampling2D((2, 2))(x)

decoded = Conv2D(1, (3, 3), padding='same')(x)
```

- smaller kernel size gives more detail. for above, change second conv2d and corresponding to 3, 3 kernel size

- this keeps same dimensionality and gives pretty good compression:
```python
x =  Conv2D(64, (5, 5), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x =  Conv2D(64, (5, 5), activation='relu', padding='same')(x)
# this will help going back the original image dimensions
x = UpSampling2D((2, 2))(x)

decoded = Conv2D(1, (3, 3), padding='same')(x)
```
