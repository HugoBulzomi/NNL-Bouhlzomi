import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.image as mpimg
from matplotlib import cm
import matplotlib.pyplot as plt
import random


def loadImages(path):
    # On charge les images, les met dans une liste
    image_files = sorted([os.path.join(path, file) for file in os.listdir(path)])
    return [np.array(cv2.imread(i, cv2.IMREAD_UNCHANGED))[:, :, :3]/255 for i in image_files]



class_names = ["no_mask", "mask"]

img_height = 128
img_width = 128
class0_path = "..\\img\\bounding_boxes_photos\\no_mask"
class1_path = "..\\img\\bounding_boxes_photos\\mask"

class0_imgs = loadImages(class0_path)
class1_imgs = loadImages(class1_path)


# On mélange notre dataset
random.shuffle(class0_imgs)
random.shuffle(class1_imgs)
num_class0 = len(class0_imgs)
num_class1 = len(class1_imgs)
print("Nombre d'images de classe 0: ", num_class0)
print("Nombre d'images de classe 1: ", num_class1)


# Ici on découpe notre dataset pour l'entrainement et la validation
X_train0 = class0_imgs[:int(num_class0*0.70)]
X_train1 = class1_imgs[:int(num_class1*0.70)]
y_train = [0 for i in range(len(X_train0))] + [1 for i in range(len(X_train1))]
X_train = X_train0 + X_train1

X_test0 = class0_imgs[int(num_class0*0.70):]
X_test1 = class1_imgs[int(num_class1*0.70):]
y_test = [0 for i in range(len(X_test0))] + [1 for i in range(len(X_test1))]
X_test = X_test0 + X_test1

# On transforme nos deux datasets en tensors pour tensorflow
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))


# On choisis une taille de batch et on mélange nos datasets à l'aide de méthodes tensorflow
BATCH_SIZE = 2
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# Définition du modèle
model=tf.keras.Sequential([
        layers.Conv2D(4, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(8, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(len(class_names), activation='softmax')
    ])

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])

# Entrainement
epochs=30
history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs)
model.summary()


# On sauvegarde le modèle
model.save("saved_model/3conv_1dropout_0.5")

 
