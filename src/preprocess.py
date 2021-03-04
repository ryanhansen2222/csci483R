import tensorflow as tf


class Preprocess():

    def __init__(self):
        #Nothing to do
        pass



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

