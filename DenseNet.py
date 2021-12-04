import tensorflow as tf
from itertools import zip_longest


class TransitionLayer(tf.keras.Model):
    def __init__(self, filters):
        """
        Constructs a ....
        """
        super(TransitionLayer, self).__init__()

        self.conv = tf.keras.layers.Conv2D(filters*2, kernel_size=1)
        self.bn = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.AveragePooling2D(
            pool_size=2, strides=2, padding="same")

    def call(self, input, is_training):
        """
        Performs...
          Args:
            input <tensorflow.tensor>: our preprocessed input data, we send through our model
            is_training <bool>: variable which determines if dropout is applied
          Results:
            output <tensorflow.tensor>: the predicted output of our input data
        """
        #x = self.bn(input, training= is_training)

        x = self.conv(input)
        x = self.bn(x, training=is_training)
        x = tf.nn.relu(x)
        output = self.pool(x)

        return output


class DenseBlock(tf.keras.Model):

    def __init__(self, filters, rep):
        """
        Constructs a residual block.
         Args:
           filters <int>: number of filter to apply
           rep <int>: number of little Blocks in our big BLock
        """
        super(DenseBlock, self).__init__()
        self.little_blocks = []
        self.rep = rep
        self.little_blocks = [
            [tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(4*filters, 1, 1, "valid"),
             tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters, 3, 1, "same")]
            # tf.keras.layers.Concatenate()]
            for _ in range(rep)]

    def call(self, input, is_training):
        """
        Performs a forward step in ...
          Args:
            input <tensorflow.tensor>: our preprocessed input data, we send through our model
            is_training <bool>: variable which determines if dropout is applied
          Results:
            output <tensorflow.tensor>: the predicted output of our input data
        """
        x = input
        for i in range(self.rep):
            [bn1, conv1, bn2, conv2] = self.little_blocks[i]

            y = conv1(x)
            y = bn1(y, training=is_training)
            y = tf.nn.relu(y)
            y = conv2(y)
            y = bn2(y, training=is_training)
            y = tf.nn.relu(y)
            x = tf.concat([y, x], axis=-1)
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

    def __init__(self, filters=12, blocks=3, block_rep=[2, 3, 4]):
        """
        Constructs our DenseNet model.
         Args:
           filter <int>: number of filter for our Conv layer in the DenseNet
           blocks <int>: number of DenseBLocks in our DenseNet
           block_rep <int>: number off little Blocks in each DenseBlock
        """

        super(DenseNet, self).__init__()

        self.num_blocks = blocks
        self.block_rep = block_rep

        # feature learning
        self.first_conv = tf.keras.layers.Conv2D(
            filters, kernel_size=7, strides=1, padding="same", activation='relu')
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)

        self.blocks = [DenseBlock(filters, rep=self.block_rep[i])
                       for i in range(0, self.num_blocks)]
        self.trans_layers = [TransitionLayer(
            filters) for _ in range(self.num_blocks - 1)]

        # classification
        self.bn = tf.keras.layers.BatchNormalization()
        self.global_pool = tf.keras.layers.GlobalAvgPool2D()
        self.classify = tf.keras.layers.Dense(
            10, kernel_regularizer="l1_l2", activation='softmax')

    def call(self, inputs, is_training):
        """
        Performs a forward step in our ResNet
          Args:
            inputs: <tensorflow.tensor> our preprocessed input data, we send through our model
            is_training: <bool> variable which determines if dropout is applied
          Results:
            output: <tensorflow.tensor> the predicted output of our input data
        """
        x = self.first_conv(inputs, training=is_training)
        x = self.pool(x, training=is_training)
        #x = tf.nn.relu(x)

        for i in range(self.num_blocks):
            x = self.blocks[i](x, is_training)
            if i != self.num_blocks-1:
                x = self.trans_layers[i](x, is_training)

        x = self.bn(x, training=is_training)
        x = tf.nn.relu(x)
        x = self.global_pool(x, training=is_training)
        output = self.classify(x, training=is_training)

        return output
