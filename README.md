# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

An autoencoder, an unsupervised artificial neural network, is trained to replicate its input as output. It encodes the input into a lower-dimensional representation before decoding it back into the original image. The primary objective of an autoencoder is to generate an output that closely resembles the input. To achieve this, autoencoders utilize layers such as MaxPooling, convolutional, and upsampling to reduce noise in the image. In this experiment, we employ the MNIST Dataset, a collection of handwritten digits ranging from 0 to 9. Each digit is represented by a 28x28 pixel image, with a total of 60,000 samples. Our goal is to construct a convolutional neural network model capable of accurately classifying these handwritten digits into their respective numerical values.

## Convolution Autoencoder Network Model

Include the neural network model diagram.

## DESIGN STEPS

### Step1:
Begin by importing the required libraries and accessing the dataset.
### Step2:
Load the dataset and normalize its values to facilitate computational processes.
### Step3:
Introduce random noise to the images within both the training and testing sets.
### Step3:
Construct the Neural Model incorporating:
-Convolutional Layer
-Pooling Layer
-Up Sampling Layer
-Ensure that the model's input and output shapes match.
### Step4:
Validate the model using the test data manually.
### Step5:
Visualize the predictions by generating plots.
## PROGRAM
```
Name: Rakshitha devi J
Reg no:212221230082

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()
x_train.shape
x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))

noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape)
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

input_img = keras.Input(shape=(28, 28, 1))

x=layers.Conv2D(16,(5,5),activation='relu',padding='same')(input_img)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(4,(3,3),activation='relu',padding='same')(x)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(4,(3,3),activation='relu',padding='same')(x)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(8,(7,7),activation='relu',padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

x=layers.Conv2D(4,(3,3),activation='relu',padding='same')(encoded)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(4,(3,3),activation='relu',padding='same')(x)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(8,(5,5),activation='relu',padding='same')(x)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(16,(5,5),activation='relu',padding='same')(x)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(16,(5,5),activation='relu')(x)
x=layers.UpSampling2D((1,1))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)

autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=3,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))

print("Rithiga Sri.B 212221230083")
metrics = pd.DataFrame(autoencoder.history.history)
metrics[['loss','val_loss']].plot()

decoded_imgs = autoencoder.predict(x_test_noisy)
print("Rithiga Sri.B 212221230083")
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display noisy
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/Rakshithadevi/convolutional-denoising-autoencoder/assets/94165326/fe6f49b9-046b-44f6-b948-53a522031b0f)


### Original vs Noisy Vs Reconstructed Image

![image](https://github.com/Rakshithadevi/convolutional-denoising-autoencoder/assets/94165326/38691a6b-05f6-4db8-9bcb-f514ea957c01)



## RESULT
Thus we have successfully developed a convolutional autoencoder for image denoising application.
