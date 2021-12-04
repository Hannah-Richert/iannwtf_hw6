import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds


def load_data():
    """
    Loading and preprocessing the data.
        Returns:
          - train_ds <tensorflow.python.data.ops.dataset_ops.PrefetchDataset>: our training dataset
          - valid_ds <tensorflow.python.data.ops.dataset_ops.PrefetchDataset>: our validation dataset
          - test_ds <tensorflow.python.data.ops.dataset_ops.PrefetchDataset>: our test dataset
    """
    # train = 50.000 and test = 10.000 images
    train_ds, valid_ds, test_ds, = tfds.load(name='cifar10', split=['train[0%:80%]','train[80%:100%]','test'], shuffle_files=True,as_supervised=True)

    train_ds = preprocess(train_ds)
    valid_ds = preprocess(valid_ds)
    test_ds = preprocess(test_ds)

    return train_ds, valid_ds, test_ds


def preprocess(ds):
    """
    Preparing our data for our model.
      Args:
        - ds <tensorflow.python.data.ops.dataset_ops.PrefetchDataset>: the dataset we want to preprocess

      Returns:
        - ds <tensorflow.python.data.ops.dataset_ops.PrefetchDataset>: preprocessed dataset
    """

    # cast labels to int32 for one hot encoding
    ds = ds.map(lambda feature, target: (tf.cast(feature, tf.float32), tf.cast(target, tf.int32)))

    # normalize images and make one-hot-encode labels
    ds = ds.map(lambda feature, target: (feature/255., tf.one_hot(target, 10)))
    # cast everything to float32
    ds = ds.map(lambda feature, target: (tf.cast(feature, tf.float32), tf.cast(target, tf.float32)))

    # cache
    ds = ds.cache()
    # shuffle, batch, prefetch our dataset
    ds = ds.shuffle(10,000)
    ds = ds.batch(32)
    ds = ds.prefetch(20)
    return ds


def train_step(model, input, target, loss_function, optimizer, is_training):
    """
    Performs a forward and backward pass for  one dataponit of our training set
      Args:
        - model <tensorflow.keras.Model>: our created MLP model
        - input <tensorflow.tensor>: our input
        - target <tensorflow.tensor>: our target
        - loss_funcion <keras function>: function we used for calculating our loss
        - optimizer <keras function>: our optimizer used for backpropagation

      Returns:
        - loss <float>: our calculated loss for the datapoint
      """

    with tf.GradientTape() as tape:

        # forward step
        prediction = model(input,is_training)

        # calculating loss
        loss = loss_function(target, prediction)

        # calculaing the gradients
        gradients = tape.gradient(loss, model.trainable_variables)

    # updating weights and biases
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def test(model, test_data, loss_function, is_training):
    """
    Test our MLP, by going through our testing dataset,
    performing a forward pass and calculating loss and accuracy
      Args:
        - model <tensorflow.keras.Model>: our created MLP model
        - test_data <tensorflow.python.data.ops.dataset_ops.PrefetchDataset> our preprocessed test dataset
        - loss_funcion <keras function>: function we used for calculating our loss

      Returns:
          - loss <float>: our mean loss for this epoch
          - accuracy <float>: our mean accuracy for this epoch
    """

    # initializing lists for accuracys and loss
    accuracy_aggregator = []
    loss_aggregator = []

    for (input, target) in test_data:

        # forward step
        prediction = model(input,is_training)

        # calculating loss
        loss = loss_function(target, prediction)

        # add loss and accuracy to the lists
        loss_aggregator.append(loss.numpy())
        for t, p in zip(target, prediction):
            accuracy_aggregator.append(tf.reduce_mean(tf.cast(tf.math.argmax(t) == tf.math.argmax(p), tf.float32)))

    # calculate the mean of the loss and accuracy (for this epoch)
    loss = tf.reduce_mean(loss_aggregator)
    accuracy = tf.reduce_mean(accuracy_aggregator)

    return loss, accuracy


def visualize(train_losses, valid_losses, valid_accuracies):
    """
    Displays the losses and accuracies from the different models in a plot-grid.
      Args:
        - train_losses- <list>: mean training losses per epoch
        - valid_losses <list>: mean testing losses per epoch
        - valid_accuracies <list>: mean accuracies (testing dataset) per epoch
    """

    fig, axs = plt.subplots(2,1)

    axs[0].plot(train_losses)
    axs[0].plot(valid_losses)
    axs[1].plot(valid_accuracies)
    axs[1].sharex(axs[0])

    fig.legend([" Train_ds loss", " Valid_ds loss", " Valid_ds accuracy"])
    plt.xlabel("Training epoch")
    fig.tight_layout()
    plt.show()
