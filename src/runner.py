import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt


from preprocess import Preprocess
from network import Network
from plotter import Plotter



if __name__ == '__main__':

    '''
    data = Preprocess().preprocess()
    (train_images, train_labels), (test_images, test_labels) = data.load_data()
    '''
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


    params = ['sequential', 3, [28*28, 128, 10], 'relu']
    network = Network().make_network(params)
    network.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])



    #TRAIN MODEL
    print('Training Learner')
    network.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = network.evaluate(test_images,  test_labels, verbose=2)
    print('\nTraining Data accuracy:', test_acc)

    print('Done Training')


    #MAKE PREDICTIONS
    print('Making Predictions')
    probability_model = tf.keras.Sequential([network,
                                             tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)
    print('Done Making Predictions')

    #MAKE PLOTS
    print('Plotting')
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    Plotter().show_img(0, predictions, test_labels, test_images, class_names)



    print('Done')
