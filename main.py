import tensorflow as tf
from tensorflow.python.ops.nn_ops import dropout
from util import load_data, test, visualize
from SimpleModel import MyModel
import numpy as np
from classify import classify
from receptive_field import ReceptiveField
from ResNet import ResNet
from DenseNet import DenseNet

tf.keras.backend.clear_session()

train_ds, valid_ds, test_ds = load_data()
optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

models = [MyModel(),ResNet(2),DenseNet(2)]

# calculating receptive field
#rfield = ReceptiveField()
# rfield for positon (cell) (0,0) in the output "image"
#rfield.add_rfield([0, 0])
# get the first image from the dataset
#img = next(iter(next(iter(next(iter(train_ds))))))
#rfield.compute_rfield_from_layers(model.get_layers(), [28, 28])
#rfield.plot(tf.cast(img,tf.int32).numpy(), (8, 8), (0, 0))


with tf.device('/device:gpu:0'):
    #x_in = tf.keras.layers.Input(shape= (32,32,3))
    #_,trained_m1 = classify(model,optimizer,1,x_in,x_in)
    #_,trained_m2 = classify(model2,optimizer,1,x_in,x_in)
    #_,trained_m3 = classify(model3,optimizer,1,x_in,x_in)
    # training the model
    for model in models:
        results, trained_model = classify(model, optimizer, 10, train_ds, valid_ds)

        # testing the trained model
        # (this code snippet should only be inserted when one decided on all hyperparameters)
        #_, test_accuracy = test(trained_model, test_ds,tf.keras.losses.CategoricalCrossentropy(),False)
        #print("Accuracy (test set):", test_accuracy)

        # visualizing losses and accuracy
        visualize(results[0],results[1],results[2])
