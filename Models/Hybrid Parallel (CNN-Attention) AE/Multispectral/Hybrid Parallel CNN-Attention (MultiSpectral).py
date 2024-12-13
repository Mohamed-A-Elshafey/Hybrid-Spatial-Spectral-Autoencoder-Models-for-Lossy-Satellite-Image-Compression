# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 12:48:24 2023

@author: Dr Badr
"""

import numpy as np
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import sys

#%%
#**Load EuroSat dataset and split the data into train 70%, validation 20%, test 10%**

ds_train, ds_val, ds_test = tfds.load('eurosat/all',
                               split=['train[:70%]', 'train[70%:90%]', 'train[90%:]'],
                               as_supervised = True)

#We need to use mapping to axis only the images since the dataset has lots of information
#in addition to labelling

train = ds_train.map(lambda image, label: image)
val = ds_val.map(lambda image, label: image)
test = ds_test.map(lambda image, label: image)

# make the dataset "images" iterable
train_np = tfds.as_numpy(train)
val_np = tfds.as_numpy(val)
test_np = tfds.as_numpy(test)

# iterate over the images and save it to a list
x_train = []
x_val = []
x_test = []

for ex in train_np:
  #print (ex.shape)
  x_train.append(ex)

for ex in val_np:
  #print (ex.shape)
  x_val.append(ex)

for ex in test_np:
  #print (ex.shape)
  x_test.append(ex)

 # Convert List to array
x_train = np.array(x_train)
x_val = np.array(x_val)
x_test = np.array(x_test)

x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
rows, cols, channels = ex.shape
print(ex.shape)


#%%
# Hybrid parallel (CNN-LSTM) Autoencoder archeticture full compressor
#rows, cols, channels = 64,64,13
input_img = keras.Input(shape=(rows, cols, channels))

###############################################################################
# Spectral encoder path with attention mechanism for spectral feature extraction
forwardSpectralFlattened = layers.Flatten()(input_img)
forwardSpectral = layers.Reshape([64, rows * channels])(forwardSpectralFlattened)
attention = layers.Attention()([forwardSpectral, forwardSpectral, forwardSpectral])
forwardSpectral = layers.Reshape([rows, cols, channels])(attention)
forwardSpectral = layers.Conv2D(7, (3, 3), activation='PReLU', padding='same')(forwardSpectral)
print("forwardSpectral", forwardSpectral.shape)

##################################################################
#spatial encoder path for spatial feature extraction
forwardSacial = input_img
print ("\nforwardSacial before: ", forwardSacial.shape)
forwardSacial = layers.Conv2D(64, (3, 3), activation='PReLU', padding='same')(forwardSacial)
forwardSacial = layers.Conv2D(64, (3, 3), activation='PReLU', padding='same')(forwardSacial)
forwardSacial = layers.Conv2D(6, (3, 3), activation='PReLU', padding='same')(forwardSacial)
print ("forwardSacial after: ", forwardSacial.shape)

###############################################################
#Concatenate
Concatenated = layers.Concatenate(axis = 3)([forwardSpectral, forwardSacial])
print ("\nConcatenated : ",Concatenated.shape)

###############################################################################
#encoder Path cont.
encoded = layers.Conv2D(64, (3, 3), activation='PReLU', padding='same')(Concatenated)
encoded = layers.MaxPooling2D((2, 2), padding='same')(encoded)

encoded = layers.Conv2D(64, (3, 3), activation='PReLU', padding='same')(encoded)
encoded = layers.MaxPooling2D((2, 2), padding='same')(encoded)

encoded = layers.Conv2D(64, (3, 3), activation='PReLU', padding='same')(encoded)
encoded = layers.MaxPooling2D((2, 2), padding='same')(encoded)



###############################################################################
#decoder Path

decoded = layers.Conv2D(64, (3, 3), activation='PReLU', padding='same')(encoded)
decoded = layers.UpSampling2D((2, 2))(decoded)

decoded = layers.Conv2D(64, (3, 3), activation='PReLU', padding='same')(decoded)
decoded = layers.UpSampling2D((2, 2))(decoded)

decoded = layers.Conv2D(64, (3, 3), activation='PReLU', padding='same')(decoded)
decoded = layers.UpSampling2D((2, 2))(decoded)
decoded = layers.Conv2D(6, (3, 3), activation='PReLU', padding='same')(decoded)
print ("decoded : ",decoded.shape)

#################################
# Spectral decoder path with attention mechanism
backwardSpectralFlattened = layers.Flatten()(decoded)
backwardSpectral = layers.Reshape([64, int(rows * channels)])(backwardSpectralFlattened)
attention = layers.Attention()([backwardSpectral, backwardSpectral, backwardSpectral])
backwardSpectral = layers.Reshape([rows, cols, channels])(attention)
backwardSpectral = layers.Conv2D(7, (3, 3), activation='PReLU', padding='same')(backwardSpectral)
print ("\nbackwardSpectral : ",backwardSpectral.shape)
##################################################################
#spatial_1 decoder path
backwardSacial = decoded
print ("\nbackwardSacial before: ",backwardSacial.shape)
backwardSacial = layers.Conv2D(64, (3, 3), activation='PReLU', padding='same')(backwardSacial)
backwardSacial = layers.Conv2D(64, (3, 3), activation='PReLU', padding='same')(backwardSacial)
backwardSacial = layers.Conv2D(6, (3, 3), activation='PReLU', padding='same')(backwardSacial)
print ("backwardSacial after: ",backwardSacial.shape)

output = layers.Concatenate(axis = 3)([backwardSpectral, backwardSacial])
print("\noutput: ", output.shape)

autoencoder = keras.Model(input_img, output)
#autoencoder.compile(optimizer='adam', loss='BinaryCrossentropy', metrics=['accuracy'])

#autoencoder.compile(optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-06),
#                     loss='mse', metrics=['accuracy'])
autoencoder.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001),
                   loss='mse', metrics=['accuracy'])
autoencoder.summary()


#Train using train and validation sets
history = autoencoder.fit(x_train, x_train,
                epochs=500,
                batch_size=64,
                shuffle=True,
                validation_data=(x_val, x_val),
                verbose = 1,
                #callbacks=[tensorboard_callback]
                )

#plot loss and accuracy
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("mse")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(['train','test'], loc = 'upper right')
plt.show

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(['train','test'], loc = 'lower right')
plt.show


#%%
#save model
autoencoder.save('autoencoderattention_MSmodel.h5')

#load model
model = load_model('autoencoderattention_MSmodel.h5')


#%%
# serialize model to JSON
model_json = autoencoder.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
autoencoder.save_weights("model_weights.h5")
print("Saved model to disk")
 
# Loading
 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_weights.h5")
print("Loaded model from disk")


#Predict the recostructed images from testing data
reconstructed = autoencoder.predict(x_test)

import numpy as np
from tensorflow.keras.preprocessing import image
############ CALULATE THE INFERENCE TIME ##############
# Load the image
img = autoencoder.predict(np.expand_dims(x_test[0], axis=0)) #REPLACE WITH LOADED MODEL NAME
 
_,r,c,ch = img.shape
img = np.reshape(img, (r,c,ch))
reconstructed.append(img)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize if necessary
        
        
import time

start_time = time.time()
predictions = autoencoder.predict(img_array)
end_time = time.time()

inference_time = end_time - start_time
print(f"Inference time: {inference_time:.4f} seconds")

# Calculate FLOPs for a Conv2D layer
def calculate_conv2d_flops(layer):
    kernel_height, kernel_width = layer.kernel_size
    in_channels = layer.input_shape[-1]
    out_channels = layer.output_shape[-1]
    flops_per_element = kernel_height * kernel_width * in_channels
    flops_per_output = flops_per_element * out_channels
    total_flops = flops_per_output * np.prod(layer.output_shape[1:-1])
    return total_flops

# Calculate MACs for a Conv2D layer
def calculate_conv2d_macs(layer):
    total_flops = calculate_conv2d_flops(layer)
    return total_flops * 2  # Assuming one MAC for each FLOP

# Calculate FLOPs for an LSTM layer
def calculate_lstm_flops(layer):
    units = layer.units
    time_steps = layer.input_shape[1]
    flops_per_input = units * (2 * layer.input_shape[-1] + units - 1)
    total_flops = flops_per_input * time_steps
    return total_flops

# Calculate MACs for an LSTM layer
def calculate_lstm_macs(layer):
    total_flops = calculate_lstm_flops(layer)
    return total_flops * 2  # Assuming one MAC for each FLOP

# Calculate total FLOPs and MACs for the model
total_flops = 0
total_macs = 0

for layer in autoencoder.layers:
    if isinstance(layer, keras.layers.Conv2D):
        total_flops += calculate_conv2d_flops(layer)
        total_macs += calculate_conv2d_macs(layer)
    elif isinstance(layer, keras.layers.LSTM):
        total_flops += calculate_lstm_flops(layer)
        total_macs += calculate_lstm_macs(layer)

# Calculate computation time (randomly chosen value)
#computation_time = 0.001  # 1 ms

# Calculate GFLOPS and GMAC
gflops = (total_flops / inference_time) * 1e-9
gmac = (total_macs / inference_time) * 1e-9

print("Total GFLOPS:", gflops)
print("Total GMAC:", gmac)

import numpy as np

def normalize_image(image):
    # Normalize the image to the range [0, 1]
    image = (image - image.min()) / (image.max() - image.min())
    return image

for i in range(0, 20):
    normalized_x_test = normalize_image(x_test[i])
    normalized_reconstructed = normalize_image(reconstructed[i])
    test_value = PSNR(normalized_x_test, normalized_reconstructed)
    print(f"PSNR test value no. {i+1} = {test_value} dB")


for i in range (0,20):
    normalized_x_test = normalize_image(x_test[i])
    normalized_reconstructed = normalize_image(reconstructed[i])
    ssim_skimage = SSIM(normalized_x_test, normalized_reconstructed, multichannel=True, data_range=x_test.max() - x_test.min())
    print(f"SSIM SKIMAGE no. {i+1} = {ssim_skimage}")
for i in range (0,20):
    normalized_x_test = normalize_image(x_test[i])
    normalized_reconstructed = normalize_image(reconstructed[i])
    mu_ssim_skimage = SSIM(normalized_x_test, normalized_reconstructed, data_range=x_test.max() - x_test.min(), win_size=11, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, multichannel=True)
    print(f"Multiscale SSIM SKIMAGE no. {i+1} = {mu_ssim_skimage}")

import matplotlib.pyplot as plt

for i in range(5):
    # Display grayscale images
    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    axes[0].imshow(x_test[i][:, :, 0], cmap='gray')
    axes[0].set_title(f"Original Image {i+1}")
    axes[1].imshow(reconstructed[i][:, :, 0], cmap='gray')
    axes[1].set_title(f"Reconstructed Image {i+1}")
    plt.show()

for i in range(5):
    # Display RGB composite images
    test_image = x_test[i][:, :, [2, 1, 0]]  # Create an RGB composite using bands 4, 2, and 1
    reconstructed_image = reconstructed[i][:, :, [2, 1, 0]]  # Create an RGB composite using bands 4, 2, and 1

    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    axes[0].imshow(test_image)
    axes[0].set_title(f"Original Image {i+1}")
    axes[1].imshow(reconstructed_image)
    axes[1].set_title(f"Reconstructed Image {i+1}")
    plt.show()
