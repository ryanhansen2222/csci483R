import tensorflow as tf



class Network():

    #Params is the list of input vars for the learners.
    #the first one is always the name
    def __init__(self):
        #Do nothing
        pass




    def make_network(self, params):
        #Do routine based on learner choice

        print('Generating Learner')

        network = None
        if(params[0] == 'sequential'):
            network = self.sequential(params)


        print('Finished Generating Learner')
        return network


    def sequential(self, params):
        numlayers = params[1]
        nodesforlayers = params[2]
        activation = params[3]


        layers = [tf.keras.layers.Flatten(input_shape=(28,28))]
        for x in range(1, numlayers):
            layers.append(tf.keras.layers.Dense(nodesforlayers[x], activation=activation))

        network = tf.keras.Sequential(layers)
        return network




