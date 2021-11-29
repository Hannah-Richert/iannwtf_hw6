import tensorflow as tf
from util import train_step, test


def classify(model, optimizer, num_epochs, train_ds, valid_ds):
    """
    Trains and tests our predefined model.
        Args:
            - model: <tensorflow.keras.Model> our untrained model
            - optimizer: <keras function> optimizer for the model
            - num_epochs: <int> number of training epochs
            - train_ds: <tensorflow.python.data.ops.dataset_ops.PrefetchDataset> our training dataset
            - valid_ds: <tensorflow.python.data.ops.dataset_ops.PrefetchDataset> our validation set for testing and optimizing hyperparameters
        Returns:
            - results: <list<list<float>>> list with losses and accuracies
            - model: <tensorflow.keras.Model> our trained MLP model
    """

    tf.keras.backend.clear_session()

    # initialize the loss: categorical cross entropy
    cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()

    # initialize lists for later visualization.
    train_losses = []
    valid_losses = []
    valid_accuracies = []

    # testing on our valid_ds once before we begin
    valid_loss, valid_accuracy = test(model, valid_ds, cross_entropy_loss,False)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)

    # Testing on our train_ds once before we begin
    train_loss, _ = test(model, train_ds, cross_entropy_loss,False)
    train_losses.append(train_loss)

    # training our model for num_epochs epochs.
    for epoch in range(num_epochs):
        print(
            f'Epoch: {str(epoch+1)} starting with (validation set) accuracy {valid_accuracies[-1]} and loss {valid_losses[-1]}')

        # training (and calculating loss while training)
        epoch_loss_agg = []

        for input, target in train_ds:
            train_loss = train_step(
                model, input, target, cross_entropy_loss, optimizer,True)
            epoch_loss_agg.append(train_loss)

        # track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))
        print(f'Epoch: {str(epoch+1)} train loss: {train_losses[-1]}')

        # testing our model in each epoch to track accuracy and loss on the validation set
        valid_loss, valid_accuracy = test(model, valid_ds, cross_entropy_loss,False)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)
    model.summary()
    results = [train_losses, valid_losses, valid_accuracies]
    return results, model
