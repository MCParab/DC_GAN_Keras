import os 
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt 

from keras.layers import Input 
from keras.models import Model, Sequential 
from keras.layers.core import Reshape, Dense, Dropout, Flatten 
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.datasets import mnist 
from keras.optimizers import Adam 
from keras import backend as K 
from keras import initializers 

K.set_image_dim_ordering('th')

#Deterministic output 
np.random.seed(1000)

randomDim = 100 

# Load MNIST Data 
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train[:, np.newaxis, :, :]

# Optimizer 
adam = Adam(lr = 0.0002, beta_1 = 0.5 )

# Generator 

# Discriminator 

# Combined Network 


# Losses 
dlosses = []
glosses = []

# Plot the losses from each batch 
def plotLoss(epoch):
	plt.figure(figsize=(10,8))
	plt.plot(dlosses, label = "D Loss")
	plt.plot(glosses, label = "G Loss")
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.lagend()
	plt.savefig('images/dcgan_loss_epoch_%d.png' % epoch)


# Wall of generated images 
def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10,10)):
	noise = np.random.normal(0, 1, size = [examples, randomDim])
	generatedImages = generator.predict(noise)

	plt.figure(figsize=figsize)
	for i in range(generatedImages.shape[0]):
		plt.subplot(dim[0], dim[1], i+1)
		plt.imshow(generatedImages[i, 0], interpolation = 'nearest', cmap = 'gray_r')
		plt.axis('off')
	plt.tight_layout()
	plt.savefig('images/dcgan_generated_image_epoch_%d.png' % epoch)

# Save the generator and discriminator networks (and weights) 
def saveModels(epoch):
	generator.save('model/dcgan_generator_epoch_%d.h5' % epoch)
	dicriminator.save('models/dcgan_discriminator+epoch_%d.h5' % epoch)

# Training .................
