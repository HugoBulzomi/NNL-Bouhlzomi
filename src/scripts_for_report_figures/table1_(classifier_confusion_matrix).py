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


classifier = tf.keras.models.load_model("../saved_model/3conv_1dropout_0.5")

class_names = ["no_mask", "mask"]

img_height = 128
img_width = 128
class0_path = "..\\..\\img\\bounding_boxes_photos\\no_mask"
class1_path = "..\\..\\img\\bounding_boxes_photos\\mask"

class0_imgs = loadImages(class0_path)
class1_imgs = loadImages(class1_path)


random.shuffle(class0_imgs)
random.shuffle(class1_imgs)
num_class0 = len(class0_imgs)
num_class1 = len(class1_imgs)
print("Nombre d'images de classe 0: ", num_class0)
print("Nombre d'images de classe 1: ", num_class1)


# Ici on découpe notre dataset pour l'entrainement et la validation

X_test0 = class0_imgs
X_test1 = class1_imgs
y_test = [0 for i in range(len(X_test0))] + [1 for i in range(len(X_test1))]
X_test = X_test0 + X_test1


predictions = np.argmax(classifier.predict(np.array(X_test)), axis=1)

mat = tf.math.confusion_matrix(predictions, y_test)
print("CONFUSION MATRIX : ", mat)
mat = mat.numpy()
acc = (mat[0][0]+mat[1][1])/(mat[0][0]+mat[1][1]+mat[1][0]+mat[0][1])
rec = (mat[1][1])/(mat[1][1]+mat[1][0])
prec = (mat[1][1])/((mat[1][1]+mat[0][1]))
f1 = 2*(prec*rec/(prec+rec))
print("accuracy: ", acc)
print("precision: ", prec)
print("recall: ", rec)
print("F1: ", f1)
