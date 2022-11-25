"""
Title: Pneumonia Classification on TPU
Author: Amy MiHyun Jang
Date created: 2020/07/28
Last modified: 2020/08/24
Description: Medical image classification on TPU.

## Introduction + Set-up

This tutorial will explain how to build an X-ray image classification model to predict 
whether an X-ray scan shows presence of pneumonia.

## Load the data

The Chest X-ray data we are using from the following weblink that divides the data into
training and test files. Let's first load in the training TFRecords.
[*Cell*](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5) 

We need a Google Cloud link to our data to load the data using a TPU. Below, we define 
key configuration parameters we'll use in this example. To run on TPU, this example 
must be on Colab with the TPU runtime selected.

Let's count how many healthy/normal chest X-rays we have and how many pneumonia chest 
X-rays we have:

Notice there are way more images that are classified as pneumonia than normal. This 
shows that we have an imbalance in our data. We will correct for this imbalance later 
on in our notebook.

We want to map each filename to the corresponding (image, label) pair. The following
methods will help us do that.

As we only have two labels, we will encode the label so that `1` or `True` indicates
pneumonia and `0` or `False` indicates normal.

## Visualize the dataset

First, let's use buffered prefetching so we can yield data from disk without having I/O
become blocking.

Please note that large image datasets should not be cached in memory. We do it here
because the dataset is not very large and we want to train on TPU.

As the method takes in NumPy arrays as its parameters, call the numpy function on the
batches to return the tensor in NumPy array form.

## Build the CNN

To make our model more modular and easier to understand, let's define some blocks. As we
are building a convolution neural network, we'll create a convolution block and a dense
layer block.

The architecture for this CNN has been inspired by this [article]
(https://towardsdatascience.com/deep-learning-for-detecting-pneumonia-from-x-ray-images-fc9a3d9fdba8).

The following method will define the function to build our model for us.

The images originally have values that range from [0, 255]. CNNs work better with smaller
numbers so we will scale this down for our input.

The Dropout layers are important, as they reduce the likelikhood of the model overfitting. 
We want to end the model with a `Dense` layer with one node, as this will be the binary 
output that determines if an X-ray shows presence of pneumonia.

## Correct for data imbalance

We saw earlier in this example that the data was imbalanced, with more images classified
as pneumonia than normal. We will correct for that by using class weighting:

The weight for class `0` (Normal) is a lot higher than the weight for class `1` (Pneumonia), 
because there are less normal images, each normal image will be weighted more to balance 
the data as the CNN works best when the training data is balanced.

## Train the model

# Defining callbacks

The checkpoint callback saves the best weights of the model, so next time we want to use
the model, we do not have to spend time training it. The early stopping callback stops
the training process when the model starts becoming stagnant, or even worse, when the
model starts overfitting.

We also want to tune our learning rate. Too high of a learning rate will cause the model
to diverge. Too small of a learning rate will cause the model to be too slow. We implement 
the exponential learning rate scheduling method below.

# Fit the model

For our metrics, we want to include precision and recall as they will provide use with a
more informed picture of how good the model is. Accuracy tells us what fraction of the
labels is correct. Since our data is not balanced, accuracy might give a skewed sense of
a good model (i.e. a model that always predicts PNEUMONIA will be 74% accurate but is not
a good model).

Precision is the number of true positives (TP) over the sum of TP and false positives (FP). 
It shows what fraction of labeled positives are actually correct.

Recall is the number of TP over the sum of TP and false negatves(FN). It shows what fraction 
of actual positives are correct.

Since there are only two possible labels for the image, we will be using the binary cross-
entropy loss. When we fit the model, remember to specify the class weights, which we defined 
earlier. Because we are using a TPU, training will be quick - less than 2 minutes.

## Visualize model performance

We plot the model accuracy and loss for the training and the validating set. Note no random 
seed is specified for this notebook. For your notebook, there might be slight  variance. We 
see the accuracy for our model is around 95%.

## Predict and evaluate results

# Evaluate the model on the test data!

We see that our accuracy on the test data is lower than the accuracy for the validating set. 
This may indicate overfitting.

Our recall is greater than our precision, indicating that almost all pneumonia images are
correctly identified but some normal images are falsely identified. We should aim to increase 
our precision.

## Google GCE Error Message

The script must be complied with Google TPU and it also need to be adjusted to adapt to the 
Google GCE Certificate. Otherwise, it will show the error message as follows:
 
All attempts to get a Google authentication bearer token failed, returning an empty token. 
Retrieving token from files failed with "NOT_FOUND: Could not locate the credentials file.". 
Retrieving token from GCE failed with "FAILED_PRECONDITION: Error executing an HTTP request: 
libcurl code 6 meaning 'Couldn't resolve host name', error details: Could not resolve host: 
metadata".
"""


import re
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    print("Device:", tpu.master())
    strategy = tf.distribute.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print("Number of replicas:", strategy.num_replicas_in_sync)


AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 25 * strategy.num_replicas_in_sync
IMAGE_SIZE = [180, 180]
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


## Load the data

train_images = tf.data.TFRecordDataset(
    "gs://download.tensorflow.org/data/ChestXRay2017/train/images.tfrec"
)
train_paths = tf.data.TFRecordDataset(
    "gs://download.tensorflow.org/data/ChestXRay2017/train/paths.tfrec"
)

ds = tf.data.Dataset.zip((train_images, train_paths))


COUNT_NORMAL = len(
    [
        filename
        for filename in train_paths
        if "NORMAL" in filename.numpy().decode("utf-8")
    ]
)
print("Normal images count in training set: " + str(COUNT_NORMAL))

COUNT_PNEUMONIA = len(
    [
        filename
        for filename in train_paths
        if "PNEUMONIA" in filename.numpy().decode("utf-8")
    ]
)
print("Pneumonia images count in training set: " + str(COUNT_PNEUMONIA))


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, "/")
    # The second to last is the class-directory
    return parts[-2] == "PNEUMONIA"


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size.
    return tf.image.resize(img, IMAGE_SIZE)


def process_path(image, path):
    label = get_label(path)
    # load the raw data from the file as a string
    img = decode_img(image)
    return img, label


ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)


# Split the data into a training and validation datasets.
ds = ds.shuffle(10000)
train_ds = ds.take(4200)
val_ds = ds.skip(4200)


# Visualize the shape of an (image, label) pair.
for image, label in train_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())


# Load and format the test data as well.
test_images = tf.data.TFRecordDataset(
    "gs://download.tensorflow.org/data/ChestXRay2017/test/images.tfrec"
)
test_paths = tf.data.TFRecordDataset(
    "gs://download.tensorflow.org/data/ChestXRay2017/test/paths.tfrec"
)
test_ds = tf.data.Dataset.zip((test_images, test_paths))

test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)


## Visualize the dataset

def prepare_for_training(ds, cache=True):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


# Call the next batch iteration of the training data.

train_ds = prepare_for_training(train_ds)
val_ds = prepare_for_training(val_ds)

image_batch, label_batch = next(iter(train_ds))


# Define the method to show the images in the batch.

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n] / 255)
        if label_batch[n]:
            plt.title("PNEUMONIA")
        else:
            plt.title("NORMAL")
        plt.axis("off")


show_batch(image_batch.numpy(), label_batch.numpy())


## Build the CNN

from tensorflow import keras
from tensorflow.keras import layers


def conv_block(filters, inputs):
    x = layers.SeparableConv2D(filters, 3, activation="relu", padding="same")(inputs)
    x = layers.SeparableConv2D(filters, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.MaxPool2D()(x)

    return outputs


def dense_block(units, dropout_rate, inputs):
    x = layers.Dense(units, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dropout(dropout_rate)(x)

    return outputs


def build_model():
    inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = layers.MaxPool2D()(x)

    x = conv_block(32, x)
    x = conv_block(64, x)

    x = conv_block(128, x)
    x = layers.Dropout(0.2)(x)

    x = conv_block(256, x)
    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x)
    x = dense_block(512, 0.7, x)
    x = dense_block(128, 0.5, x)
    x = dense_block(64, 0.3, x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


## Correct for data imbalance

initial_bias = np.log([COUNT_PNEUMONIA / COUNT_NORMAL])
print("Initial bias: {:.5f}".format(initial_bias[0]))

TRAIN_IMG_COUNT = COUNT_NORMAL + COUNT_PNEUMONIA
weight_for_0 = (1 / COUNT_NORMAL) * (TRAIN_IMG_COUNT) / 2.0
weight_for_1 = (1 / COUNT_PNEUMONIA) * (TRAIN_IMG_COUNT) / 2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print("Weight for class 0: {:.2f}".format(weight_for_0))
print("Weight for class 1: {:.2f}".format(weight_for_1))


## Train the model

# Defining callbacks
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("xray_model.h5", save_best_only=True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)


initial_learning_rate = 0.015
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)


# Fit the model
with strategy.scope():
    model = build_model()

    METRICS = [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="binary_crossentropy",
        metrics=METRICS,
    )

history = model.fit(
    train_ds,
    epochs=100,
    validation_data=val_ds,
    class_weight=class_weight,
    callbacks=[checkpoint_cb, early_stopping_cb],
)


## Visualizing model performance

fig, ax = plt.subplots(1, 4, figsize=(20, 3))
ax = ax.ravel()

for i, met in enumerate(["precision", "recall", "binary_accuracy", "loss"]):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history["val_" + met])
    ax[i].set_title("Model {}".format(met))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(met)
    ax[i].legend(["train", "val"])


## Predict and evaluate results

# Evaluate the model on our test data!
model.evaluate(test_ds, return_dict=True)

for image, label in test_ds.take(1):
    plt.imshow(image[0] / 255.0)
    plt.title(CLASS_NAMES[label[0].numpy()])

prediction = model.predict(test_ds.take(1))[0]
scores = [1 - prediction, prediction]

for score, name in zip(scores, CLASS_NAMES):
    print("This image is %.2f percent %s" % ((100 * score), name))