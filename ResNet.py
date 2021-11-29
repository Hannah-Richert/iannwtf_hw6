import tensorflow as tf

class ResidualBlock(tf.keras.Model):

    def __init__(self,n_filters=[32,32,32]):
        """
        Constructs a residual block.
        """
        super(ResidualBlock, self).__init__()

        [filters1,filters2,filters3] = n_filters

        self.conv1 = tf.keras.layers.Conv2D(filters = filters1, kernel_size = 1, strides=1,padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters = filters2, kernel_size = 3, strides=1,padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2D(filters = filters3, kernel_size = 1, strides=1,padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, is_training=False):
        """
        Performs a forward step in our Residual block
          Args:
            inputs: <tensorflow.tensor> our preprocessed input data, we send through our model
            is_training: <bool> variable which determines if dropout is applied
          Results:
            output: <tensorflow.tensor> the predicted output of our input data
        """

        x = self.conv1(inputs)
        x = self.bn1(x,training = is_training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x,training = is_training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x,training = is_training)

        x += inputs
        output = tf.nn.relu(x)

        return output


class ResNet(tf.keras.Model):

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

        super(ResNet, self).__init__()
        # feature learning
        self.first_conv = tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, strides=1,padding="same",activation='relu')
        self.blocks = [ResidualBlock(n_filters=[32,32,32]) for i in range(num_blocks)]
        # classification
        self.global_pool = tf.keras.layers.GlobalAvgPool2D()
        self.classify = tf.keras.layers.Dense(10, kernel_regularizer="l1_l2", activation='softmax')


    def call(self, input,is_training):
        """
        Performs a forward step in our ResNet.
          Args:
            input: <tensorflow.tensor> our preprocessed input data, we send through our model
            is_training: <bool> variable which determines if dropout is applied
          Results:
            output: <tensorflow.tensor> the predicted output of our input data
        """
        x = self.first_conv(input,training = is_training)

        for block in self.blocks:
            x = block(x, is_training)

        x = self.global_pool(x,training = is_training)
        output = self.classify(x,training = is_training)

        return output
