"""
Title: Consistency training with supervision
Author: [Sayak Paul](https://twitter.com/RisingSayak)
Date created: 2021/04/13
Last modified: 2021/04/19
Description: Training with consistency regularization for robustness against data  
distribution shifts.

Deep learning models excel in image recognition tasks when the data is independent and 
identically distributed (i.i.d.). However, they can suffer from performance degradation 
caused by subtle distribution shifts in the input data (such as random noise, contrast 
change, and blurring). So, naturally, there arises a question of why. As discussed in 
[A Fourier Perspective on Model Robustness in Computer Vision]
(https://arxiv.org/pdf/1906.08988.pdf)), there's no reason for deep learning models to 
be robust against such shifts. Standard model training procedures(such as standard image 
classification training workflows) don't enable a model to learn beyond what is fed to 
it in the form of training data.

In this example, we train an image classification model enforcing a sense of consistency 
inside it by doing the following:

* Train a standard image classification model.
* Train an _equal or larger_ model on a noisy version of the dataset (augmented using
  [RandAugment](https://arxiv.org/abs/1909.13719)).
* To do this, we will first obtain predictions of the previous model on the clean images
  of the dataset.
* Use these predictions and train the second model to match these predictions on the 
  noisy variant of the same images. This is identical to the workflow of [*Knowledge 
  Distillation*](https://keras.io/examples/vision/knowledge_distillation/) but since 
  the student model is equal or larger in size this process is also referred to as 
  Self-Training.

This overall training workflow finds its roots in the works as follows. 

[FixMatch](https://arxiv.org/abs/2001.07685),
[Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848), 
[Noisy Student Training](https://arxiv.org/abs/1911.04252). 

Since this training process encourages a model yield consistent predictions for clean and
noisy images, it's often referred to as consistency training or training with consistency 
regularization. Although the example focuses on using consistency training to enhance the 
robustness of models to common corruptions this example can serve a template to perform
 _weakly supervised learning_.

The example requires TensorFlow 2.4 or higher, TensorFlow Hub and TensorFlow Models. Users 
have three options to use the package of tf-models-official in case any one method could 
not be failed. 

Option 1. Istall the packakge by shell (in Miniconda)

$ conda intall pip
$ pip install -q tf-models-official 
$ pip install -q tensorflow-addons

Import: 
from official.vision.image_classification.augment import RandAugment

Option 2. Copy models and add image_classification

1).Copy the models 
Copy the models from TF2.10 to the current directory such as the miniconda environment
/home/user/miniconda3/lib/python3.9/site-packages/tensorflow/models

TF2.10 Models Without image_classification 
https://github.com/tensorflow/models/tree/master/official/vision

2).Copy image_classification into the above-mentioned models into the directory of vision
TF 2.8 With image_classification
https://github.com/tensorflow/models/tree/v2.8.0/official/vision

The final path is exampled as follows. 
/home/user/miniconda3/lib/python3.9/site-packages/tensorflow/models/official/vision/image_classification

import: 
from tensorflow.models.official.vision.image_classification.augment import RandAugment

Option 3. Directly adopt augment script

1).Copy augment.py from TF2.8 to current working directory

2).Adopt the following import
from augment import RandAugment

The major procedures are listed as follows. 

## Create TensorFlow Dataset objects

To train the teacher model, we only use two geometric augmentation transforms: random
 horizontal flip and random crop.

To enable train_clean_ds and train_noisy_ds are shuffled using the same seed to ensure 
their orders are exactly the same. It is helpful during training the student model.

## Define a model building utility function

To define our model building utility. The model is based on the [ResNet50V2 architecture]
(https://arxiv.org/abs/1603.05027).

In the interest of reproducibility, we serialize the initial random weights of the teacher 
network.

## Train the teacher model

As noted in Noisy Student Training, it leads to better performance if the teacher model 
is trained with geometric ensembling and when the student model is forced to mimic that,  
The original work uses [Stochastic Depth](https://arxiv.org/abs/1603.09382) and [Dropout]
(https://jmlr.org/papers/v15/srivastava14a.html) to bring in the ensemblingpart but for 
the example, we use [Stochastic Weight Averaging](https://arxiv.org/abs/1803.05407)(SWA)
which also resembles geometric ensembling.

## Define a self-training utility

For this part, we will borrow the Distiller class from [this Keras Example]
(https://keras.io/examples/vision/knowledge_distillation/).
# Majority of the code is taken from:
# https://keras.io/examples/vision/knowledge_distillation/

The only difference in this implementation is the way that the  loss is being calculated. 
Instead of weighted the distillation loss and student loss differently that we are taking 
their average following Noisy Student Training.

## Assess the robustness of the models

A standard benchmark of assessing to the robustness of vision models is to record their 
performance on corrupted datasets such as ImageNet-C and CIFAR-10-C that were proposed 
in [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations]
(https://arxiv.org/abs/1903.12261). For this example, we will be using the CIFAR-10-C 
dataset which has 19 different corruptions on 5 different severity levels. To assess 
the robustness of the models on this dataset, we will do the following:

* Run the pre-trained models on the highest level of severities and obtain the top-1 
  accuracies.
* Compute the mean top-1 accuracy.

For the purpose of this example, we will not be going through these steps. This is why 
we trained the models for only 5 epochs. You can carefully check out [this repository]
(https://github.com/sayakpaul/Consistency-Training-with-Supervision) that demonstrates 
the full-scale training experiments and also the aforementioned assessment. The figure 
below presents an executive summary of that assessment:

![](https://i.ibb.co/HBJkM9R/image.png)

Mean Top-1 results stand for the CIFAR-10-C dataset and Test Top-1 results stand for 
the CIFAR-10 test set. It's clear that consistency training has an advantage on not
only enhancing the model robustness but also on improving the standard test performance.
"""


# -from official.vision.image_classification.augment import RandAugment
from tensorflow.models.official.vision.image_classification.augment import RandAugment
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

tf.random.set_seed(42)


## Define hyperparameters

AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 128
EPOCHS = 5

CROP_TO = 72
RESIZE_TO = 96


## Load the CIFAR-10 dataset

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

val_samples = 49500
new_train_x, new_train_y = x_train[: val_samples + 1], y_train[: val_samples + 1]
val_x, val_y = x_train[val_samples:], y_train[val_samples:]


## Create TensorFlow Dataset objects

# Initialize RandAugment object.
augmenter = RandAugment(num_layers=2, magnitude=9)


def preprocess_train(image, label, noisy=True):
    image = tf.image.random_flip_left_right(image)
    # We first resize the original image to a larger dimension
    # and then we take random crops from it.
    image = tf.image.resize(image, [RESIZE_TO, RESIZE_TO])
    image = tf.image.random_crop(image, [CROP_TO, CROP_TO, 3])
    if noisy:
        image = augmenter.distort(image)
    return image, label


def preprocess_test(image, label):
    image = tf.image.resize(image, [CROP_TO, CROP_TO])
    return image, label


train_ds = tf.data.Dataset.from_tensor_slices((new_train_x, new_train_y))
validation_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# The dataset will be used to train the first model.
train_clean_ds = (
    train_ds.shuffle(BATCH_SIZE * 10, seed=42)
    .map(lambda x, y: (preprocess_train(x, y, noisy=False)), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

# This prepares the Dataset object to use RandAugment.
train_noisy_ds = (
    train_ds.shuffle(BATCH_SIZE * 10, seed=42)
    .map(preprocess_train, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

validation_ds = (
    validation_ds.map(preprocess_test, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

test_ds = (
    test_ds.map(preprocess_test, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

# The dataset will be used to train the second model.
consistency_training_ds = tf.data.Dataset.zip((train_clean_ds, train_noisy_ds))


## Visualize the datasets

sample_images, sample_labels = next(iter(train_clean_ds))
plt.figure(figsize=(10, 10))
for i, image in enumerate(sample_images[:9]):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy().astype("int"))
    plt.axis("off")

sample_images, sample_labels = next(iter(train_noisy_ds))
plt.figure(figsize=(10, 10))
for i, image in enumerate(sample_images[:9]):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy().astype("int"))
    plt.axis("off")


## Define a model building utility function

def get_training_model(num_classes=10):
    resnet50_v2 = tf.keras.applications.ResNet50V2(
        weights=None,
        include_top=False,
        input_shape=(CROP_TO, CROP_TO, 3),
    )
    model = tf.keras.Sequential(
        [
            layers.Input((CROP_TO, CROP_TO, 3)),
            layers.Rescaling(scale=1.0 / 127.5, offset=-1),
            resnet50_v2,
            layers.GlobalAveragePooling2D(),
            layers.Dense(num_classes),
        ]
    )
    return model


initial_teacher_model = get_training_model()
initial_teacher_model.save_weights("initial_teacher_model.h5")


## Train the teacher model

# Define the callbacks.
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=3)
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)

# Initialize SWA from tf-hub.
SWA = tfa.optimizers.SWA

# Compile and train the teacher model.
teacher_model = get_training_model()
teacher_model.load_weights("initial_teacher_model.h5")
teacher_model.compile(
    # Note that we are wrapping the optimizer within SWA
    optimizer=SWA(tf.keras.optimizers.Adam()),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
history = teacher_model.fit(
    train_clean_ds,
    epochs=EPOCHS,
    validation_data=validation_ds,
    callbacks=[reduce_lr, early_stopping],
)

# Evaluate the teacher model on the test set.
_, acc = teacher_model.evaluate(test_ds, verbose=0)
print(f"Test accuracy: {acc*100}%")


## Define a self-training utility

class SelfTrainer(tf.keras.Model):
    def __init__(self, student, teacher):
        super(SelfTrainer, self).__init__()
        self.student = student
        self.teacher = teacher

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        temperature=3,
    ):
        super(SelfTrainer, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.temperature = temperature

    def train_step(self, data):
        # Since our dataset is a zip of two independent datasets, 
        # after initially parsing them, we segregate the respective 
        # images and labels next.
        clean_ds, noisy_ds = data
        clean_images, _ = clean_ds
        noisy_images, y = noisy_ds

        # Forward pass of teacher
        teacher_predictions = self.teacher(clean_images, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(noisy_images, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            total_loss = (student_loss + distillation_loss) / 2

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in compile()
        self.compiled_metrics.update_state(
            y, tf.nn.softmax(student_predictions, axis=1)
        )

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"total_loss": total_loss})
        return results

    def test_step(self, data):
        # During inference, we only pass a dataset consisting images and labels.
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Update the metrics
        self.compiled_metrics.update_state(y, tf.nn.softmax(y_prediction, axis=1))

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        return results


## Train the student model

# Define the callbacks. We are using a larger decay factor to stabilize the training.
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    patience=3, factor=0.5, monitor="val_accuracy"
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True, monitor="val_accuracy"
)

# Compile and train the student model.
self_trainer = SelfTrainer(student=get_training_model(), teacher=teacher_model)
self_trainer.compile(
    # Notice we are not using SWA here.
    optimizer="adam",
    metrics=["accuracy"],
    student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=tf.keras.losses.KLDivergence(),
    temperature=10,
)
history = self_trainer.fit(
    consistency_training_ds,
    epochs=EPOCHS,
    validation_data=validation_ds,
    callbacks=[reduce_lr, early_stopping],
)


# Evaluate the student model.
acc = self_trainer.evaluate(test_ds, verbose=0)
print(f"Test accuracy from student model: {acc*100}%")