import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.image as mpimg
from matplotlib import cm
import matplotlib.pyplot as plt
import random

# On charge nos modèles entrainés
classifier = tf.keras.models.load_model("../saved_model/3conv_1dropout_0.5")
extractor = tf.keras.models.load_model("../saved_model/face_extractor")

def loadImages(path):
    image_files = sorted([os.path.join(path, file) for file in os.listdir(path)])
    return [np.array(cv2.resize(cv2.imread(i, cv2.IMREAD_UNCHANGED), (300, 300), interpolation = cv2.INTER_AREA))[:, :, :3]/255 for i in image_files]

class_names = ["no_mask", "mask"]

img_height = 128
img_width = 128
class0_path = "../../img/photos_nomask"
class1_path = "../../img/photos_mask"

# On charge les photos originales. On a copié et trié les photos à la main dans deux dossier exprès pour obtenir cette matrice !
class0_imgs = loadImages(class0_path)
class1_imgs = loadImages(class1_path)
random.shuffle(class0_imgs)
random.shuffle(class1_imgs)

X_test0 = class0_imgs
X_test1 = class1_imgs
y_test = [0 for i in range(len(X_test0))] + [1 for i in range(len(X_test1))]
X_test = X_test0 + X_test1

# On fait prédire tous les cadre de visage à notre modèle d'extraction
rectangles = extractor.predict(np.array(X_test))

# On rogne tout ce qui ne fais pas partie du cadre prédit pour ne garder que les images de visages. On les redimensionne ensuite en 128x128
for i in range(len(X_test)):
    X_test[i] = X_test[i][int(rectangles[i][1]):int(rectangles[i][1]+rectangles[i][3]), int(rectangles[i][0]):int(rectangles[i][0]+rectangles[i][2])]
    X_test[i] = cv2.resize(X_test[i], (128, 128), interpolation = cv2.INTER_AREA)

# Pour chaque visage ainsi obtenu, on prédit sa catégorie
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
