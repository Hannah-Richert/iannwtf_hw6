import tensorflow as tf
from tensorflow.python.ops.nn_ops import dropout
from util import load_data, test, visualize
from SimpleModel import MyModel
import numpy as np
from classify import classify
from ResNet import ResNet
from DenseNet import DenseNet

tf.keras.backend.clear_session()

train_ds, valid_ds, test_ds = load_data()
optimizer = tf.keras.optimizers.Adam(0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

# param 16,266
res_small = ResNet(block_filters = [8,8,16],out_filters=[32,32,64],blocks = 3,modes = ["n","s","n"])
# param 13,490
dense_small= DenseNet(filters=4,blocks=3,block_rep=[2,4,2], growth_rate=4)
# param
res_large = ResNet(block_filters = [32,32,64,64,128,128],out_filters=[128,128,256,256,512,512],blocks = 5,modes = ["n","s","n","s","n","n"])
# param
dense_large= DenseNet(filters=8,blocks=4,block_rep=[6,12,24,16],growth_rate=16)

# train for 10 epochs: ~60% accuracy
#models = [res_small,dense_small,MyModel()]

# train for 30 epochs: ~ 85% accuracy
models = [res_large,dense_large,MyModel()]


train_losses = []
valid_losses = []
valid_accuracies = []

with tf.device('/device:gpu:0'):
    # training the model
    for model in models:
        results, trained_model = classify(model, optimizer, 20, train_ds, valid_ds)
        trained_model.summary()
        # saving results for visualization
        train_losses.append(results[0])
        valid_losses.append(results[1])
        valid_accuracies.append(results[2])

        # testing the trained model
        # (this code snippet should only be inserted when one decided on all hyperparameters)
        _, test_accuracy = test(trained_model, test_ds,tf.keras.losses.CategoricalCrossentropy(),False)
        print("Accuracy (test set):", test_accuracy)


        # visualizing losses and accuracy
    visualize(train_losses,valid_losses,valid_accuracies)
