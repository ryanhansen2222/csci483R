import tensorflow as tf


class Preprocess():

	def __init__(self):
		#Nothing to do
		pass


	def prepgantest(self):
		(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
		train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
		train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
		BUFFER_SIZE = 60000
		BATCH_SIZE = 256
		train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
		return train_dataset

	def preprocess(self):
		print('Preprocessing Data')

		rawData = self.getData
		data = self.processData(rawData)

		print('Finished Preprocessing')
		return data


	def getData(self):
		fashion_mnist = tf.keras.datasets.fashion_mnist
		return fashion_mnist


	def processData(self, rawData):
		#Do preprocessing
		return rawData

