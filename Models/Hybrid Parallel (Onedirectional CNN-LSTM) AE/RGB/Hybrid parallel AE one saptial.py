# -*- coding: utf-8 -*-
"""
Created on Sun May 14 22:33:35 2023

@author: Dr Badr
"""

# -*- coding: utf-8 -*-
"""
Created on Tue APRIL 29 10:05:20 2023

@author: Mohamed Badr
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
from sewar.full_ref import uqi
#import sys
#from tensorflow import keras
#import visualkeras

#%%
#**Load EuroSat dataset and split the data into train 70%, validation 20%, test 10%**

ds_train, ds_val, ds_test = tfds.load('eurosat/rgb',
                               split=['train[:35%]', 'train[70%:85%]', 'train[90%:]'],
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
#rows, cols, channels = 64,64,3
input_img = keras.Input(shape=(rows, cols, channels))

###############################################################################
#sepctral encoder path for spectral feature extraction
forwardSpectralFlattened = layers.Flatten()(input_img)
forwardSpectral = layers.Reshape([64,rows*channels])(forwardSpectralFlattened)
forwardSpectral = layers.LSTM(50, return_sequences=True)(forwardSpectral)
forwardSpectral = layers.LSTM(50, return_sequences=True)(forwardSpectral)
forwardSpectral = layers.LSTM(96, return_sequences=True)(forwardSpectral)
forwardSpectral = layers.LSTM(rows*channels, return_sequences=True)(forwardSpectral)
forwardSpectral = layers.Reshape([rows, cols, channels])(forwardSpectral)
forwardSpectral = layers.Conv2D(2, (3, 3), activation='PReLU', padding='same')(forwardSpectral)
print("forwardSpectral", forwardSpectral.shape)

##################################################################
#spatial encoder path for spatial feature extraction
forwardSacial = input_img
print ("\nforwardSacial before: ", forwardSacial.shape)
forwardSacial = layers.Conv2D(64, (3, 3), activation='PReLU', padding='same')(forwardSacial)
forwardSacial = layers.Conv2D(64, (3, 3), activation='PReLU', padding='same')(forwardSacial)
forwardSacial = layers.Conv2D(1, (3, 3), activation='PReLU', padding='same')(forwardSacial)
print ("forwardSacial after: ", forwardSacial.shape)

#########################
#Concatenate
Concatenated = layers.Concatenate(axis = 3)([forwardSpectral, forwardSacial])
print ("\nConcatenated : ",Concatenated.shape) 

###############################################################################
#encoder Path cont.
encoded = layers.Conv2D(64, (3, 3), activation='PReLU', padding='same')(Concatenated)
encoded = layers.MaxPooling2D((2, 2), padding='same')(encoded)

encoded = layers.Conv2D(64, (3, 3), activation='PReLU', padding='same')(encoded)
encoded = layers.MaxPooling2D((2, 2), padding='same')(encoded)

encoded = layers.Conv2D(64, (3, 3), activation='sigmoid', padding='same')(encoded)
encoded = layers.MaxPooling2D((2, 2), padding='same')(encoded)
print ("\nencoded : ",encoded.shape) 

###############################################################################
#decoder Path

decoded = layers.Conv2D(64, (3, 3), activation='PReLU', padding='same')(encoded)
decoded = layers.UpSampling2D((2, 2))(decoded)

decoded = layers.Conv2D(64, (3, 3), activation='PReLU', padding='same')(decoded)
decoded = layers.UpSampling2D((2, 2))(decoded)

decoded = layers.Conv2D(64, (3, 3), activation='PReLU', padding='same')(decoded)
decoded = layers.UpSampling2D((2, 2))(decoded)
decoded = layers.Conv2D(3, (3, 3), activation='PReLU', padding='same')(decoded)
print ("decoded : ",decoded.shape) 

################################################################################
#sepctral decoder path
backwardSpectralFlattened = layers.Flatten()(decoded)
backwardSpectral = layers.Reshape([64,int(rows*channels)])(backwardSpectralFlattened)
backwardSpectral = layers.LSTM(96, return_sequences=True)(backwardSpectral)
backwardSpectral = layers.LSTM(50, return_sequences=True)(backwardSpectral)
backwardSpectral = layers.LSTM(50, return_sequences=True)(backwardSpectral)
backwardSpectral = layers.LSTM(rows*channels, return_sequences=True)(backwardSpectral)
backwardSpectral = layers.Reshape([rows, cols, channels])(backwardSpectral)
backwardSpectral = layers.Conv2D(2, (3, 3), activation='PReLU', padding='same')(backwardSpectral)
print ("\nbackwardSpectral : ",backwardSpectral.shape)

##################################################################
#spatial_1 decoder path
backwardSacial = decoded
print ("\nbackwardSacial before: ",backwardSacial.shape) 
backwardSacial = layers.Conv2D(64, (3, 3), activation='PReLU', padding='same')(backwardSacial)
backwardSacial = layers.Conv2D(64, (3, 3), activation='PReLU', padding='same')(backwardSacial)
backwardSacial = layers.Conv2D(1, (3, 3), activation='PReLU', padding='same')(backwardSacial)
print ("backwardSacial after: ",backwardSacial.shape)

output = layers.Concatenate(axis = 3)([backwardSpectral, backwardSacial])
print("\noutput: ", output.shape)

autoencoder = keras.Model(input_img, output)
#autoencoder.compile(optimizer='adam', loss='BinaryCrossentropy', metrics=['accuracy'])
#autoencoder.compile(optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-06), 
                   # loss='BinaryCrossentropy', metrics=['accuracy'])
autoencoder.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001), 
                    loss='mse', metrics=['accuracy'])
autoencoder.summary()

#Train using train and validation sets
history = autoencoder.fit(x_train, x_train,
                epochs=1,
                batch_size=4,
                shuffle=True,
                validation_data=(x_val, x_val),
                verbose = 1,
                #callbacks=[tensorboard_callback]
                )
 
#plot loss and accuracy
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("MSE")
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
autoencoder.save('autoencoder_model110.h5')

#load model
model = load_model('autoencoder_model110.h5')


#%%
# serialize model to JSON
model_json = autoencoder.to_json()
with open("model110.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
autoencoder.save_weights("model110_weights.h5")
print("Saved model to disk")
 
# Loading
 
# load json and create model
json_file = open('model110.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model110_weights.h5")
print("Loaded model from disk")
#Predict the recostructed images from testing data
reconstructed = autoencoder.predict(x_test)

for i in range (0,20):
        fig = plt.figure(figsize=(10, 10))
        fig.add_subplot(1, 2,1)
        plt.imshow(x_test[i])
        fig.add_subplot(1, 2,2)
        plt.imshow(reconstructed[i])
        print(f"image no. {i+1} ")
        plt.show()
        
#Claculate PSNR, SSIM, Uqi, and BPP. But Bpp is the same between testing images and reconstructed images,
# because it's the same array dimention, so we need to save it to a local disc then measure BPP   



for i in range (0,20):
        test_value = PSNR(x_test[i], reconstructed[i])
        print(f"PSNR test value no. {i+1} = {test_value} dB")
for i in range (0,20):
        universal_quality_image_index=uqi(x_test[i], reconstructed[i])
        print(f"uqi test value no. {i+1} = {universal_quality_image_index} ")
for i in range (0,20):
    ssim_skimage = SSIM(x_test[i], reconstructed[i], multichannel=True, data_range=x_test.max() - x_test.min())
    print(f"SSIM SKIMAGE no. {i+1} = {ssim_skimage}")   
for i in range (0,20):
    mu_ssim_skimage = SSIM(x_test[i], reconstructed[i], data_range=x_test.max() - x_test.min(), win_size=11, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, multichannel=True)
    print(f"Multiscale SSIM SKIMAGE no. {i+1} = {mu_ssim_skimage}")  
    
     
#%%
# Define the encoder model
encoder = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('max_pooling2d_2').output)

# Define the decoder model
decoder = tf.keras.Model(inputs=autoencoder.get_layer('conv2d_7').input, outputs=autoencoder.output) 
  
# Select one image from the test set
image_index = 0
test_image = x_test[image_index]

# Define the bpp range
bpp_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Initialize lists to store distortion scores
psnr_scores = []
ssim_scores = []
ms_ssim_scores = []

# Iterate over the bpp range
for bpp in bpp_range:
    
    # Initialize lists to store scores for each test image
    psnr_scores_per_image = []
    ssim_scores_per_image = []
    ms_ssim_scores_per_image = []

    # Iterate over test images
    #for i in range(len(x_test)):
    for i in range(0,2):
        # Select a test image
        img = x_test[i]

        # Compress the test image with the specified bpp
        compressed_img = encoder.predict(np.expand_dims(img, axis=0)) / bpp
        
        # Decompress the image
        decompressed_img = decoder.predict(compressed_img)
        _,r,c,ch = decompressed_img.shape
        decompressed_img = np.reshape(decompressed_img, (r,c,ch))
        
        # Resize the test image to match the shape of the decompressed image
        #resized_test_image = cv2.resize(img, (decompressed_img.shape[2], decompressed_img.shape[1]))   
        
        # Calculate PSNR, SSIM, and MS-SSIM
        psnr = PSNR(img, decompressed_img)
        ssim = SSIM(img, decompressed_img, multichannel=True)
        ms_ssim = SSIM(img, decompressed_img, data_range=x_test.max() - x_test.min(), win_size=11, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, multichannel=True)

        # Append scores to the per-image lists
        psnr_scores_per_image.append(psnr)
        ssim_scores_per_image.append(ssim)
        ms_ssim_scores_per_image.append(ms_ssim)

    # Calculate average scores for the bpp
    avg_psnr = np.mean(psnr_scores_per_image)
    avg_ssim = np.mean(ssim_scores_per_image)
    avg_ms_ssim = np.mean(ms_ssim_scores_per_image)

    # Append average scores to the overall lists
    psnr_scores.append(avg_psnr)
    ssim_scores.append(avg_ssim)
    ms_ssim_scores.append(avg_ms_ssim)

    print(f"BPP: {bpp:.2f}")
    print(f"Avg PSNR: {avg_psnr:.2f}")
    print(f"Avg SSIM: {avg_ssim:.4f}")
    print(f"Avg MS-SSIM: {avg_ms_ssim:.4f}")

# Save the model
autoencoder.save('autoencoder_model.h5')

# Plot rate distortion curves
plt.figure(figsize=(10, 6))
plt.plot(bpp_range, psnr_scores, marker='o', label='PSNR')
plt.xlabel('Rate(BPP)')
plt.ylabel('PSNR(dB)')
plt.title('PSNR vs BPP')
plt.grid(True)
plt.legend()
plt.ylim(25, 50)  # Set the y-axis range
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(bpp_range, ssim_scores, marker='o', label='SSIM')
plt.xlabel('Rate(BPP)')
plt.ylabel('SSIM')
plt.title('SSIM vs BPP')
plt.grid(True)
plt.legend()
plt.ylim(0.60, 1.00)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(bpp_range, ms_ssim_scores, marker='o', label='MS-SSIM')
plt.xlabel('Rate(BPP)')
plt.ylabel('MS-SSIM')
plt.title('MS-SSIM vs BPP')
plt.grid(True)
plt.legend()
plt.ylim(0.60, 1.00)
plt.show()
