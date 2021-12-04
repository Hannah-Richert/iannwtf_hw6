import tensorflow as tf
from itertools import zip_longest

class TransitionLayer(tf.keras.Model):
    def __init__(self,filters):
        """
        Constructs a ....
        """
        super(TransitionLayer, self).__init__()

        self.conv = tf.keras.layers.Conv2D(filters*2, kernel_size=1)
        self.bn = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.AveragePooling2D(pool_size = 2, strides=2,padding="same")

    @tf.function
    def call(self, input,is_training):
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


class Block(tf.keras.Model):

    def __init__(self,filters,rep):
        """
        Constructs a residual block.
         Args:
           filters <int>: number of filter to apply
           rep <int>: number of little Blocks in our big BLock
        """
        super(Block, self).__init__()
        self.little_blocks =[]
        self.rep = rep
        self.little_blocks =[DenseBlock(filters) for _ in range(self.rep)]

    @tf.function
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
        for block in self.little_blocks:
            y = block(x, is_training)
            x = tf.concat([y,x],axis=-1)

        output = x
        return output

class DenseBlock(tf.keras.Model):

    def __init__(self,filters):
        """
        Constructs a residual block.
         Args:
           filters <int>: number of filter to apply
           rep <int>: number of little Blocks in our big BLock
        """
        super(DenseBlock, self).__init__()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(4*filters,1,1,"valid")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters,3,1,"same")

    @tf.function
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
        y = self.conv1(x)
        y = self.bn1(y, training = is_training)
        y = tf.nn.relu(y)
        y = self.conv2(y)
        y = self.bn2(y, training = is_training)
        y = tf.nn.relu(y)
        output = y
        return output

class DenseNet(tf.keras.Model):

    """
    Our own custon MLP model, which inherits from the keras.Model class
      Functions:
        init: constructor of our model
        get_layer: returns list with our layers
        call: performs forward pass of our model
    """

    def __init__(self,filters=12,blocks=3,block_rep=[2,3,4],growth_rate=4):
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
        self.growth_rate = growth_rate
        # feature learning
        self.first_conv = tf.keras.layers.Conv2D(32, kernel_size = 7, strides=2,padding="valid",use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2, padding="valid")

        self.units = []
        for i in range(self.num_blocks):
            self.units.append(Block(filters, rep=self.block_rep[i]))
            if i != self.num_blocks-1:
                self.units.append(TransitionLayer(filters=self.block_rep[i]*self.growth_rate))


        # classification
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.global_pool = tf.keras.layers.GlobalAvgPool2D()
        self.classify = tf.keras.layers.Dense(10, kernel_regularizer="l1_l2", activation='softmax')

    @tf.function
    def call(self, input,is_training):
        """
        Performs a forward step in our ResNet
          Args:
            input: <tensorflow.tensor> our preprocessed input data, we send through our model
            is_training: <bool> variable which determines if dropout is applied
          Results:
            output: <tensorflow.tensor> the predicted output of our input data
        """
        x = self.first_conv(input,training = is_training)
        x = self.bn1(x,training=is_training)
        x = tf.nn.relu(x)
        x = self.pool(x)

        for unit in self.units:
            x = unit(x, is_training)

        x = self.bn2(x,training = is_training)
        x = tf.nn.relu(x)
        x = self.global_pool(x,training = is_training)
        output = self.classify(x,training = is_training)

        return output
