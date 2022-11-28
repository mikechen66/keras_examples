"""
Title: Image classification from scratch
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/04/27
Last modified: 2020/04/28
Description: Training an image classifier from scratch on the Kaggle Cats vs Dogs dataset.

## Introduction

This example shows how to do image classification from scratch, starting from JPEG image 
files on disk, without leveraging pre-trained weights or a pre-made Keras Application model. 
We demonstrate the workflow on the Kaggle Cats vs Dogs binary classification dataset.

We use the 'image_dataset_from_directory' utility to generate the datasets, and use Keras 
image preprocessing layers for image standardization and data augmentation.

## Load the data: the Cats vs Dogs dataset

Download the 786M ZIP archive of the raw data:

$ curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
$ unzip -q kagglecatsanddogs_5340.zip
$ ls PetImages

We have a 'PetImages' folder which contain two subfolders, 'Cat' and 'Dog'. Each subfolder 
contains image files for each category. You can choose to save it in a dataset folder. 

Filter out corrupted images

When working with lots of real-world image data, corrupted images are a common occurence. 
Filter out badly-encoded images that do not feature the string "JFIF" in their header.

## Visualize the data

Here are the first 9 images in the training dataset. As you can see, label 1 is "dog" and label 
0 is "cat".

## Use image data augmentation

When you don't have a large image dataset, it's a good practice to artificially introduce sample 
diversity by applying random realistic transformations to the training images, such as random 
horizontal flipping or small random rotations. This helps expose the model to different aspects 
of the training data while slowing down overfitting.

Please visualize what the augmented samples look like, by applying 'data_augmentation' repeatedly 
to the first image in the dataset.

## Standardize the data

Our image are already in a standard size(180x180), as being yielded as contiguous 'float32' batches 
by our dataset. However, their RGB channel values are in the '[0, 255]' range. This is not ideal 
for a neural network; in general you should seek to make your input values small. Here, we will
standardize values to be in the '[0, 1]'' by using a 'Rescaling' layer at the start of the model.

## Two options to preprocess the data

There are two ways you could be using the 'data_augmentation' preprocessor:

Option 1(GPU): Make it part of the model:

inputs = keras.Input(shape=input_shape)
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
...  # Rest of the model

With the option, data augmentation will happen on device, synchronously with the rest of  the model 
execution, meaning that it will benefit from GPU acceleration.Note data augmentation is inactive at 
test time, so the input samples will only be augmented during 'fit()'', not when calling 'evaluate()'
or 'predict()'. If you're training on GPU, this may be a good option.

Option 2(CPU): apply it to the dataset to yield batches of augmented images

augmented_train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y))

With the option, your data augmentation will happen on CPU, asynchronously, and will be buffered before 
going into the model. If you're training on CPU, this is the better option, since it makes data 
augmentation asynchronous and non-blocking. In our case, we'll go with the second option. If you're 
not sure which one to pick, this second option (asynchronous preprocessing) is always a solid choice.

## Configure the dataset for performance

Let's apply data augmentation to our training dataset, and let's make sure to use buffered prefetching 
so we can yield data from disk without having I/O becoming blocking:

## Build a model

We'll build a small version of the Xception network. We haven't particularly tried tooptimize the archi-
tecture; if you want to do a systematic search for the best model configuration, consider using
[KerasTuner](https://github.com/keras-team/keras-tuner).Please note the following:

-Start the model with the data_augmentation preprocessor, followed by a Rescaling layer.
-Include a Dropout layer before the final classification layer.

## Run inference on new data

Note that data augmentation and dropout are inactive at inference time. And It has the following TypeError: 
unsupported format string passed to numpy.ndarray.__format__

https://github.com/open-mmlab/mmselfsup/issues/175
"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


## Load the data: the Cats vs Dogs dataset

import os

num_skipped = 0
for folder_name in ("Cat", "Dog"):
    # folder_path = os.path.join("PetImages", folder_name)
    folder_path = os.path.join("/home/mike/datasets/kagglecatsanddogs_5340/PetImages", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print("Deleted %d images" % num_skipped)


## Generate a `Dataset`

image_size = (180, 180)
# batch_size = 128
batch_size = 32

train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    # "PetImages",
    "/home/mike/datasets/kagglecatsanddogs_5340/PetImages",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)


## Visualize the data

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")


## Using image data augmentation

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)


plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")


## Configure the dataset for performance

# Apply data_augmentation to the training images.
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)

# Prefetch samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# train_ds = train_ds.prefetch(buffer_size=32)
# val_ds = val_ds.prefetch(buffer_size=32)


## Build the model

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)


# Apply 'data_augmentation' to the training images.
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)
# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)


## Train the model

# Get to ~96% validation accuracy after training for 25 epochs on the full dataset.
# epochs = 25
epochs = 1

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
    # jit_compile=True,  # Disable XLA compilation to avoid no compilor error
)

model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)