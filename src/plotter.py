import matplotlib.pyplot as plt
import numpy as np



class Plotter():


    def __init__(self):
        pass

    def show_img(self, i, predictions, test_labels, test_images, class_names):
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        self.plot_image(i, predictions[i], test_labels, test_images, class_names)
        plt.subplot(1,2,2)
        self.plot_value_array(i, predictions[i],  test_labels)
        plt.show()


    def plot_image(self, i, predictions_array, true_label, img, class_names):
        true_label, img = true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                             100*np.max(predictions_array),
                                             class_names[true_label]),
                   color=color)

    def plot_value_array(self, i, predictions_array, true_label):
        true_label = true_label[i]
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')
