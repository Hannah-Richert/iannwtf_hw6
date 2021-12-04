import tensorflow as tf

class ResidualBlock(tf.keras.Model):
    """
    Our own custon ResidualBlock, which inherits from the keras.Model class
      Functions:
        init: constructor of our block
        call: performs forward pass of our block
    """

    def __init__(self,num_filters,out_filters,mode):
        """
        Constructs a residual block, with 3 conv layers, 2 of those are bottleneck layers.
            Args:
                num_filters <int>: number of filters for the first 2 conv-layers
                out_filters <int>: number of filters for the last layer
                mode <str>: the type of our ResBlock (n= normal/constant,s = strided)
        """
        super(ResidualBlock, self).__init__()

        self.num_filters = num_filters
        self.out_filters = out_filters
        self.mode = mode

        # bottleneck layer
        self.conv1 = tf.keras.layers.Conv2D(num_filters, kernel_size = 1, strides=1,padding="valid")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

        # middle conv layer with kernel=(3,3)
        if mode == 's':
            self.conv2 = tf.keras.layers.Conv2D(num_filters, kernel_size = 3, strides=2, kernel_regularizer="l1_l2",padding="same")
            # pooling layer for reshaping the input
            self.pool = tf.keras.layers.MaxPool2D(pool_size=1,strides=2)
        # mode: normal, constant
        else:
            self.conv2 = tf.keras.layers.Conv2D(num_filters, kernel_size = 3, strides=1, kernel_regularizer="l1_l2",padding="same")

        # bottleneck layer
        self.conv3 = tf.keras.layers.Conv2D(out_filters, kernel_size = 1, strides=1, padding="valid")
        self.bn3 = tf.keras.layers.BatchNormalization()

        # layer we might need to reshape the input
        self.conv_resize_input = tf.keras.layers.Conv2D(out_filters, kernel_size = 1, strides=1,padding="same")


    @tf.function(experimental_relax_shapes=True)
    def call(self, input, is_training):
        """
        Performs a forward step in our ResidualBlock.
          Args:
            input <tensorflow.tensor>: our preprocessed input data, we send through our model
            is_training <bool>: variable which determines if dropout is applied
          Results:
            output <tensorflow.tensor>: the predicted output of our input data
        """
        x = self.conv1(input)
        x = self.bn1(x,training = is_training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x,training = is_training)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = self.bn3(x,training = is_training)


        # depending on the mode and filters we have to reshape the input

        #[.,x,x,.] x changed by pool
        if (self.mode == 's'):
            input = self.pool(input)

        if (self.num_filters != self.out_filters):
            input = self.conv_resize_input(input)

        # adding input and processed tensor (x)
        x += input

        x = tf.nn.relu(x)

        output = x
        return output


class ResNet(tf.keras.Model):

    """
    Our own custon ResNet model, which inherits from the keras.Model class
      Functions:
        init: constructor of our model
        call: performs forward pass of our model
    """

    def __init__(self,block_filters,out_filters,blocks,modes):
        """
        Constructs our ResNet model.
        Args:
            block_filters <list<int>>: number of filters of each block
            out_filters <list<int>>: number of filters for output of each block
            blocks <int>: number of ResBlocks for our model
            modes <list<str>>: mode of each ResBlock (n= normal/constant,s = strided)
        """

        super(ResNet, self).__init__()

        # feature learning
        self.first_conv = tf.keras.layers.Conv2D(filters = 32, kernel_size = 7, strides=2, kernel_regularizer="l1_l2" ,padding="same")
        self.bn = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.MaxPool2D(pool_size = 3,strides = 2)

        self.blocks = [ResidualBlock(block_filters[i] ,out_filters[i], modes[i]) for i in range(blocks)]

        # classification
        self.global_pool = tf.keras.layers.GlobalAvgPool2D()
        self.classify = tf.keras.layers.Dense(10, kernel_regularizer="l1_l2", activation='softmax')

    @tf.function
    def call(self, input,is_training):
        """
        Performs a forward step in our ResNet.
          Args:
            input <tensorflow.tensor>: our preprocessed input data, we send through our model
            is_training <bool>: variable which determines if sertein functions are applied
          Results:
            output <tensorflow.tensor>: the predicted output of our input data
        """
        x = self.first_conv(input)
        x = self.bn(x,training=is_training)
        x = tf.nn.relu(x)
        x = self.pool(x)

        # residual blocks
        for block in self.blocks:
            x = block(x, is_training)

        # classification
        x = self.global_pool(x)
        output = self.classify(x)

        return output
