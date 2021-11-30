import tensorflow as tf

class ResidualBlock(tf.keras.Model):

    def __init__(self,num_filters=32,out_filters=32):
        """
        Constructs a residual block.
            Args:
                num_filters <int>: number of filters for the first 2 conv-layers
                out_filters <int>: number of filters for the last layer
        """
        super(ResidualBlock, self).__init__()

        self.num_filters = num_filters
        self.out_filters = out_filters

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size = 1, strides=1,padding="same")

        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size = 3, strides=1,padding="same")

        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters = out_filters, kernel_size = 1, strides=1,padding="same")

        self.conv_resize_input = tf.keras.layers.Conv2D(filters = out_filters, kernel_size = 1, strides=1,padding="same")


    def call(self, input, is_training=False):
        """
        Performs a forward step in our ResidualBlock.
          Args:
            input <tensorflow.tensor>: our preprocessed input data, we send through our model
            is_training <bool>: variable which determines if dropout is applied
          Results:
            output <tensorflow.tensor>: the predicted output of our input data
        """
        x = self.bn1(input,training = is_training)
        x = tf.nn.relu(x)
        x = self.conv1(x)

        x = self.bn2(x,training = is_training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        x = self.bn3(x,training = is_training)
        x = tf.nn.relu(x)
        x = self.conv3(x)

        # resizing input if input-size != processed-tensor-size (x)
        if (self.num_filters != self.out_filters):
            input = self.conv_resize_input(input)

        # adding input and processed tensor (x)
        x += input

        output = x
        return output


class ResNet(tf.keras.Model):

    """
    Our own custon ResNet model, which inherits from the keras.Model class
      Functions:
        init: constructor of our model
        get_layer: returns list with our layers
        call: performs forward pass of our model
    """

    def __init__(self,block_filters=[16,32,64],blocks=2):
        """
        Constructs our ResNet model.
        Args:
            block_filters <list<int>>: number of filters for each blocks length=blocks+1
            blocks <int>: number of ResBlocks for our model
        """

        super(ResNet, self).__init__()

        # feature learning
        self.first_conv = tf.keras.layers.Conv2D(filters = block_filters[0], kernel_size = 3, strides=1,padding="same")
        # block filters == filters from first_conv/ previous out_filters
        self.blocks = [ResidualBlock(num_filters=block_filters[i] ,out_filters=block_filters[i+1]) for i in range(blocks)]

        # classification
        self.global_pool = tf.keras.layers.GlobalAvgPool2D()
        self.classify = tf.keras.layers.Dense(10, kernel_regularizer="l1_l2", activation='softmax')


    def call(self, input,is_training):
        """
        Performs a forward step in our ResNet.
          Args:
            input <tensorflow.tensor>: our preprocessed input data, we send through our model
            is_training <bool>: variable which determines if dropout is applied
          Results:
            output <tensorflow.tensor>: the predicted output of our input data
        """
        x = self.first_conv(input,training = is_training)

        for block in self.blocks:
            x = block(x, is_training)

        x = self.global_pool(x,training = is_training)
        output = self.classify(x,training = is_training)

        return output
