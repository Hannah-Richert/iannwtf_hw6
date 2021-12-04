import tensorflow as tf
from itertools import zip_longest

class TransitionLayer(tf.keras.Model):
    """
    Our own custon TransitionLayer, which inherits from the keras.Model class
      Functions:
        init: constructor of our layer
        call: performs forward pass of our layer
    """
    def __init__(self,filters):
        """
        Constructs a Transition Layer, which reduces complexity in our model (it cuts the size of our input in half).
            Args:
                filters <int>: number of filters applied to the conv2D layer
        """
        super(TransitionLayer, self).__init__()

        self.conv = tf.keras.layers.Conv2D(filters, kernel_size=1,padding="valid")
        self.bn = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.AveragePooling2D(pool_size = 2, strides=2,padding="same")

    @tf.function(experimental_relax_shapes=True)
    def call(self, input,is_training):
        """
        Performs a forward step in our TransitionLayer
          Args:
            input <tensorflow.tensor>: our preprocessed input data, we send through our model
            is_training <bool>: variable which determines if BatchNormalization is applied
          Results:
            output <tensorflow.tensor>: the predicted output of our input data
        """

        x = self.conv(input)
        x = self.bn(x, training=is_training)
        x = tf.nn.relu(x)
        output = self.pool(x)

        return output

class LittleBlock(tf.keras.Model):
    """
    Our own custon DenseBLock, which inherits from the keras.Model class
      Functions:
        init: constructor of our DenseBLock
        call: performs forward pass of our DenseBLock
    """

    def __init__(self,filters):
        """
        Constructs a dense block.
         Args:
           filters <int>: number of filters to apply
        """
        super(LittleBlock, self).__init__()

        #bottleneck layer
        self.conv1 = tf.keras.layers.Conv2D(4*filters,1,1,"valid")
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters,3,1,"same")
        self.bn2 = tf.keras.layers.BatchNormalization()

    @tf.function(experimental_relax_shapes=True)
    def call(self, input, is_training):
        """
        Performs a forward step in our DenseBlock.
        Sending the input trogh multiple Convolutional layers and a relu-activation after each Conv-layer with BatchNormaization
          Args:
            input <tensorflow.tensor>: our preprocessed input data, we send through our model
            is_training <bool>: variable which determines if dropout is applied
          Results:
            output <tensorflow.tensor>: the predicted output of our input data
        """

        x = self.conv1(input)
        x = self.bn1(x, training = is_training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training = is_training)
        x = tf.nn.relu(x)

        output = x
        return output

class DenseBlock(tf.keras.Model):
    """
    Our own custon  Dense Block, which inherits from the keras.Model class
      Functions:
        init: constructor of our block
        call: performs forward pass of our block
    """
    def __init__(self,filters,rep):
        """
        Constructs a big dense block, which consists of multiple LittleBlocks
         Args:
           filters <int>: number of filters to apply to the conv layers
           rep <int>: number of Dense Blocks in our Block
        """
        super(DenseBlock, self).__init__()
        self.blocks =[]
        self.rep = rep
        self.blocks =[LittleBlock(filters) for _ in range(self.rep)]

    @tf.function(experimental_relax_shapes=True)
    def call(self, input, is_training):
        """
        Performs a forward step of our block.
        It iterates through the DenseBocks.
        The output of a DenseBlock is concatenated with its input and becomes the Input for the next DenseBlock
          Args:
            input <tensorflow.tensor>: our preprocessed input data, we send through our model
            is_training <bool>: variable which determines if dropout is applied
          Results:
            output <tensorflow.tensor>: the predicted output of our input data
        """
        x = input
        for block in self.blocks:
            y = block(x, is_training)
            x = tf.concat([y,x],axis=-1)

        output = x
        return output


class DenseNet(tf.keras.Model):

    """
    Our own custon DenseNet model, which inherits from the keras.Model class.

      Functions:
        init: constructor of our model
        call: performs forward pass of our model
    """

    def __init__(self,filters=12,blocks=3,block_rep=[2,3,4],growth_rate=4):
        """
        Constructs our DenseNet model.
         Args:
           filter <int>: basis number of filter for our Conv layer in the DenseNet
           blocks <int>: number of large BLocks in our DenseNet
           block_rep <int>: number off DenseBlocks in each larger Block
           growth_rate <int>: how much will our models complexity will be reduced in the transition layer
        """

        super(DenseNet, self).__init__()

        self.num_blocks = blocks
        self.block_rep = block_rep
        self.growth_rate = growth_rate

        # feature learning
        self.first_conv = tf.keras.layers.Conv2D(filters = 32, kernel_size = 7, strides=2,padding="valid",use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2, padding="valid")

        self.units = []
        for i in range(self.num_blocks):
            self.units.append(DenseBlock(filters, rep=self.block_rep[i]))
            if i != self.num_blocks-1:
                self.units.append(TransitionLayer(filters= self.block_rep[i]*self.growth_rate))


        # classification
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.global_pool = tf.keras.layers.GlobalAvgPool2D()
        self.classify = tf.keras.layers.Dense(10, kernel_regularizer="l1_l2", activation='softmax')

    @tf.function
    def call(self, input,is_training):
        """
        Performs a forward step in our ResNet.
          Args:
            input: <tensorflow.tensor> our preprocessed input data, we send through our model
            is_training: <bool> variable which determines if dropout is applied
          Results:
            output: <tensorflow.tensor> the predicted output of our input data
        """
        x = self.first_conv(input)
        x = self.bn1(x,training=is_training)
        x = tf.nn.relu(x)
        x = self.pool(x)

        # iterating through our DenseBlocks and TransitionLayers
        for unit in self.units:
            x = unit(x, is_training)

        x = self.bn2(x,training = is_training)
        x = tf.nn.relu(x)
        x = self.global_pool(x)
        output = self.classify(x)

        return output
