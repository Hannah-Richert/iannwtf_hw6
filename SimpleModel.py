import tensorflow as tf


class MyModel(tf.keras.Model):
    """
    Our own custon MLP model, which inherits from the keras.Model class
      Functions:
        init: constructor of our model
        get_layer: returns list with our layers
        call: performs forward pass of our model
    """

    def __init__(self):
        """
        Constructs our model.
        """

        super(MyModel, self).__init__()

        # feature learning
        self.l1 = tf.keras.layers.Conv2D(
            filters=12, kernel_size=5, strides=1, padding="same", activation='relu')
        self.l2 = tf.keras.layers.Conv2D(
            filters=12, kernel_size=3, strides=1, padding="same", activation='relu')
        self.l3 = tf.keras.layers.Dropout(0.2)
        self.l4 = tf.keras.layers.MaxPool2D(
            pool_size=2, strides=2, padding="same")
        self.l5 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=9, strides=1, padding="same", kernel_regularizer="l1_l2", activation='relu')
        self.l6 = tf.keras.layers.GlobalAvgPool2D()
        # classification
        self.l7 = tf.keras.layers.Dense(
            10, kernel_regularizer="l1_l2", activation='softmax')

    def get_layers(self):
        """
        Returns list with all layers.

          Returns:
            - <list<tensorflow.keras.layer>> list with the models layers
        """
        return [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7]

    def call(self, input, is_training):
        """
        Performs a forward step in our MLP
          Args:
            input: <tensorflow.tensor> our preprocessed input data, we send through our model
            is_training: <bool> variable which determines if dropout is applied
          Results:
            output: <tensorflow.tensor> the predicted output of our input data
        """

        x = self.l1(input, training=is_training)
        x = self.l2(x, training=is_training)
        x = self.l3(x, training=is_training)
        x = self.l4(x, training=is_training)
        x = self.l5(x, training=is_training)
        x = self.l6(x, training=is_training)
        output = self.l7(x, training=is_training)

        return output
