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

#small_models = [ResNet(block_filters = [16,24,32],blocks = 2),DenseNet(filters=6,blocks=2,block_rep=3)]
#big_models = [ResNet(block_filters = [8,8,8,12,12,12],out_filters=[16,16,16,24,24,24],blocks = 6),]
#models = small_models.append(MyModel())
# parameters(trainable) = 11,834 / accuracy (valid_ds; 10 epochs) = 55.9%
#models = [ResNet(block_filters = [16,24,32],blocks = 2)]
# parameters(trainable) = 34,442 / accuracy (valid_ds; 10 epochs) = 56,8%
# models = [ResNet(block_filters = [16,24,32,16,24,32], blocks = 5)]
# parameters(trainable) = 59.066 / accuracy (valid_ds; 10 epochs) = 60,7%
#models = [ResNet(block_filters = [16,32,64,32], blocks = 3)]
# parameters(trainable) = 248,522 / accuracy (valid_ds; 10 epochs) = 59,4%
#models = [ResNet(block_filters = [32,64,128,32], blocks = 3)]

models = [ResNet(block_filters = [8,8,12,24],out_filters=[16,16,24,24],modes = ["strided","normal","normal","constant"],blocks = 4)]
#models = [DenseNet(filters=4,blocks=3,block_rep=[2,6,4])]


# parameters(trainable) = 11.422  / accuracy (valid_ds; 10 epochs) = 56.9%
#models = [DenseNet(filters=6,blocks=2,block_rep=3)]
# parameters(trainable) = 25,266 / accuracy (valid_ds; 10 epochs) = 65.4%
#models = [DenseNet(filters=4, blocks=3, block_rep=8)]
# parameters(trainable) = 91,306  / accuracy (valid_ds; 10 epochs) = 64,6%
#models = [DenseNet(filters=12,blocks=3,block_rep=4)]


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
