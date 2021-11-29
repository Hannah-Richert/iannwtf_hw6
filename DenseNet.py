import tensorflow as tf
from itertools import zip_longest

class TransitionLayer(tf.keras.Model):
    def __init__(self):
        """
        Constructs a ....
        """
        super(TransitionLayer, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.conv = tf.keras.layers.Conv2D(filters = 32, kernel_size=1)
        self.pool = tf.keras.layers.AveragePooling2D(pool_size = 2, strides=2,padding="same")

    def call(self, input,is_training):
        """
        Performs...
          Args:
            input: <tensorflow.tensor> our preprocessed input data, we send through our model
            is_training: <bool> variable which determines if dropout is applied
          Results:
            output: <tensorflow.tensor> the predicted output of our input data
        """
        #x = self.bn(input, training= is_training)
        x = tf.nn.relu(input)
        x = self.conv(input)
        output = self.pool(x)

        return output


class DenseBlock(tf.keras.Model):

    def __init__(self,filters,num_rep):
        """
        Constructs a residual block.
        """
        super(DenseBlock, self).__init__()
        self.little_blocks =[]
        self.num_rep = num_rep
        self.little_blocks =[[
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(4*filters,1,1,"same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters,3,1,"same"),
            tf.keras.layers.Concatenate()] for _ in range(num_rep)]


    def call(self, input, is_training):
        """
        Performs a forward step in ...
          Args:
            input: <tensorflow.tensor> our preprocessed input data, we send through our model
            is_training: <bool> variable which determines if dropout is applied
          Results:
            output: <tensorflow.tensor> the predicted output of our input data
        """
        x = input
        for i in range(self.num_rep):
            [bn1,conv1,bn2,conv2,conc] = self.little_blocks[i]
            y = bn1(x, training = is_training)
            y = tf.nn.relu(y)
            y = conv1(y)
            y = bn2(y, training = is_training)
            y = tf.nn.relu(y)
            y = conv2(y)
            x = conc([y,x])
        output = x
        return output


class DenseNet(tf.keras.Model):

    """
    Our own custon MLP model, which inherits from the keras.Model class
      Functions:
        init: constructor of our model
        get_layer: returns list with our layers
        call: performs forward pass of our model
    """

    def __init__(self,num_blocks):
        """
        Constructs our ResNet model.
        """

        super(DenseNet, self).__init__()
        self.first_conv = tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, strides=1,padding="same",activation='relu')
        self.num_blocks = num_blocks
        # feature learning
        self.blocks = [DenseBlock(filters=32,num_rep=1) for _ in range(0,num_blocks)]
        self.trans_layers = [TransitionLayer() for _ in range(num_blocks-1)]
        # classification
        self.bn = tf.keras.layers.BatchNormalization()
        self.global_pool = tf.keras.layers.GlobalAvgPool2D()
        self.classify = tf.keras.layers.Dense(10, kernel_regularizer="l1_l2", activation='softmax')


    def call(self, inputs,is_training):
        """
        Performs a forward step in our ResNet
          Args:
            inputs: <tensorflow.tensor> our preprocessed input data, we send through our model
            is_training: <bool> variable which determines if dropout is applied
          Results:
            output: <tensorflow.tensor> the predicted output of our input data
        """
        x = self.first_conv(inputs,training = is_training)

        for i in range(self.num_blocks):
            x = self.blocks[i](x, is_training)
            if i != self.num_blocks-1:
                x = self.trans_layers[i](x, is_training)


        x = self.bn(x,training = is_training)
        x = self.global_pool(x,training = is_training)
        output = self.classify(x,training = is_training)

        return output