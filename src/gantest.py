import tensorflow as tf



import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import tensorflow_docs.vis.embed as embed

from IPython import display
from gan_test_network import Gan_Test_Network
from preprocess import Preprocess




# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))



if __name__ == '__main__':
	pre = Preprocess()
	testdata = pre.prepgantest()
	gan = Gan_Test_Network()
	gan.run(testdata)
	
	

                             


	#display_image(EPOCHS)
	#GIF
	'''
	anim_file = 'dcgan.gif'

	with imageio.get_writer(anim_file, mode='I') as writer:
		filenames = glob.glob('image*.png')
		filenames = sorted(filenames)
		for filename in filenames:
			image = imageio.imread(filename)
			writer.append_data(image)
		image = imageio.imread(filename)
		writer.append_data(image)

	embed.embed_file(anim_file)
	'''

		
	print('Done')

	
