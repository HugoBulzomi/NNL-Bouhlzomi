import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.image as mpimg
from matplotlib import cm
import matplotlib.pyplot as plt
import random

# On charge les models entrainés
classifier = tf.keras.models.load_model("saved_model/3conv_1dropout_0.5")
extractor = tf.keras.models.load_model("saved_model/face_extractor")


def predict(img_path, mode):
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if image.dtype == "uint16":
        image = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
        print("Attention, nos modèles sont entrainés avec des images de type uint8.\n Vos images sont de type {}".format(image.dtype))
    
    resized = np.array(cv2.resize(image, (300, 300), interpolation = cv2.INTER_AREA))[:, :, :3]/255

    rectangle = extractor.predict(np.array([resized]))[0]

    cropped = resized[int(rectangle[1]):int(rectangle[1]+rectangle[3]), int(rectangle[0]):int(rectangle[0]+rectangle[2])]
    cropped = cv2.resize(cropped, (128, 128), interpolation = cv2.INTER_AREA)

    pred = classifier.predict(np.array([cropped]))[0]
    #print(pred)

    if mode == "probabilities":
        print("nomask: {}\nmask: {}".format(pred[0], pred[1]))
    elif mode == "categories":
        print(["nomask", "mask"][np.argmax(pred)])
    else:
        print("mode should be 'probabilities' or 'categories'")

    #cv2.imshow('BGR Image',cropped)
    #cv2.waitKey(0)
                       

# On fait deux prédiction pour deux photos originales
predict("..\\img\\my_photos\\20220115_150828.jpg", mode="probabilities")
predict("..\\img\\my_photos\\20220115_150910.jpg", mode="categories")
