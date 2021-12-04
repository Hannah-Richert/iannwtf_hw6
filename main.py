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
optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)


model1 = [ResNet(block_filters = [8,8,12,24],out_filters=[16,16,24,24],modes = ["strided","normal","normal","constant"],blocks = 4)]
model2= [DenseNet(filters=4,blocks=3,block_rep=[2,4,3], growth_rate=4)]
models = [MyModel(),model1,model2]




with tf.device('/device:gpu:0'):
    # training the model
    for model in models:
        results, trained_model = classify(model, optimizer, 10, train_ds, valid_ds)
        # testing the trained model
        # (this code snippet should only be inserted when one decided on all hyperparameters)
        _, test_accuracy = test(trained_model, test_ds,tf.keras.losses.CategoricalCrossentropy(),False)
        print("Accuracy (test set):", test_accuracy)

        # visualizing losses and accuracy
        visualize(results[0],results[1],results[2])
