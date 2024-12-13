# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 10:59:49 2022

@author: Dr Badr
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:05:20 2022

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
from skimage.metrics import structural_similarity as ssim
from sewar.full_ref import uqi
import sys
#import visualkeras
#%% 
#**Load EuroSat dataset and split the data into train 70%, validation 20%, test 10%**

ds_train, ds_val, ds_test = tfds.load('eurosat/rgb',
                               split=['train[:40%]', 'train[70%:85%]', 'train[90%:]'],
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
#%%
# Hybrid (CNN-LSTM) cascade Autoencoder archeticture
rows, cols, channels = 64,64,3
input_img = keras.Input(shape=(rows, cols, channels))
x = layers.Conv2D(64, (3, 3), activation='PReLU', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Flatten()(x)
print (x.shape)
x = x = layers.Reshape([64,1024])(x)
x = layers.LSTM(100, return_sequences=True)(x)
x = layers.LSTM(100, return_sequences=True)(x)
x = layers.LSTM(192, return_sequences=True)(x)
print (x.shape)

###############################################################################
x = layers.Reshape([64,64,3])(x)

## DECODER ###
x = layers.Conv2D(64, (3, 3), activation='PReLU', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Flatten()(x)
print (x.shape)
x = x = layers.Reshape([64,4096])(x)
###############################################################################
#x = layers.LSTM(100, return_sequences=True)(x)
x = layers.LSTM(100, return_sequences=True)(x)
x = layers.LSTM(100, return_sequences=True)(x)
x = layers.LSTM(192, return_sequences=True)(x)
print (x.shape)

x = x = layers.Reshape([64,64,3])(x)
decoded = layers.Conv2D(channels, (3, 3), activation='sigmoid', padding='same')(x)



autoencoder = keras.Model(input_img, decoded)
#autoencoder.compile(optimizer='adam', loss='BinaryCrossentropy', metrics=['accuracy'])
autoencoder.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-06), 
                    loss='mse', metrics=['accuracy'])
autoencoder.summary()


#%%


'''
%load_ext tensorboard
import datetime

# Clear any logs from previous runs
!rm -rf ./logs/ 

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
'''
#Train using train and validation sets
history = autoencoder.fit(x_train, x_train,
                epochs=1000,
                batch_size=8,
                shuffle=True,
                validation_data=(x_val, x_val),
                verbose = 1,
                #callbacks=[tensorboard_callback]
                )
 
#plot loss and accuracy
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("BinaryCrossentropy")
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
autoencoder.save('autoencoder_model.h5')

#load model
model = load_model('autoencoder_model.h5')


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
    ssim_skimage = ssim(x_test[i], reconstructed[i], multichannel=True, data_range=x_test.max() - x_test.min())
    print(f"SSIM SKIMAGE no. {i+1} = {ssim_skimage}")   
    
    
#calculate image bit per pixel
for i in range (0,10):
    imgBpp = (sys.getsizeof(x_test[i])*8) / (rows * cols)
    reconstructedImageBPP = (sys.getsizeof(reconstructed[i])*8) / (rows * cols)
    print(f"original image bpp = {imgBpp}")
    print(f"reconstructed image bpp = {reconstructedImageBPP}")    
 
