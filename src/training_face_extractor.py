import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.image as mpimg
from matplotlib import cm
import matplotlib.pyplot as plt
import random
from PIL import Image


def loadImages(path):
    # On charge les images, les met dans une liste
    image_files = sorted([os.path.join(path, file) for file in os.listdir(path)])
    return [np.array(cv2.imread(i, cv2.IMREAD_UNCHANGED))[:, :, :3]/255 for i in image_files]



img_height = 128
img_width = 128
class0_path = "..\\img\\bounding_boxes_photos\\no_mask"
class1_path = "..\\img\\bounding_boxes_photos\\mask"

originals_photos_path = "..\\img\\my_photos"


#On charge les noms de nos images anotées
image_files = sorted([os.path.join(class0_path , file) for file in os.listdir(class0_path)])
image_files += sorted([os.path.join(class1_path , file) for file in os.listdir(class1_path)])


images_faces = {}
for i in image_files:
    split = i.split('-')
    # Pour chaque image annotée, on retrouve le nom de l'image originale, et les coordonées et dimensions du cadre
    img_name = split[0].split('\\')[-1]
    face_coord = split[2].split('x')
    face_dim = [split[3], split[4][:-4]]
    # On associe ensuite à chaque nom de photo originale, les images annotées qui en ont été extraites
    if images_faces.get(img_name) == None:
        images_faces[img_name] = [[face_coord[0], face_coord[1], face_dim[0], face_dim[1]]]
    else:
        images_faces[img_name].append([face_coord[0], face_coord[1], face_dim[0], face_dim[1]])

X_train = []
y_train = []

# On ne garde que les photos originales avec un seul visage 
for img in images_faces.keys():
    if len(images_faces[img]) == 1:
        scale = (1.0, 1.0)

        # Ici on charge les photos originales et on les redimentiones 
        try:
            image = cv2.imread(originals_photos_path+"\\"+img+'.jpg', cv2.IMREAD_UNCHANGED)
            scale = (300/image.shape[0], 300/image.shape[1])
            image = np.array(cv2.resize(image, (300, 300), interpolation = cv2.INTER_AREA))[:, :, :3]/255
            X_train.append(image)
        except:
            image = cv2.imread(originals_photos_path+"\\"+img+'.png', cv2.IMREAD_UNCHANGED)
            scale = (300/image.shape[0], 300/image.shape[1])
            image = np.array(cv2.resize(image, (300, 300), interpolation = cv2.INTER_AREA))[:, :, :3]/255
            X_train.append(image)
            
        
        face = images_faces[img][0]

        # Puisque la photo originale a été redimentionnée, on applique cette redimenssion au cadre du visage
        face[0] = int(face[0])*scale[1]
        face[1] = int(face[1])*scale[0]
        face[2] = int(face[2])*scale[1]
        face[3] = int(face[3])*scale[0]

        y_train.append(face)

# On mélange nos X_train et y_train de la même façon
c = list(zip(X_train, y_train))
random.shuffle(c)
X_train, Y_train = zip(*c)

# X_train comporte 70% du dataset, et X_test les 30% restants
X_test = np.array(X_train[:int(len(X_train)*0.3)])
y_test = np.array(y_train[:int(len(y_train)*0.3)])
X_train = np.array(X_train[int(len(X_train)*0.3):])
y_train = np.array(Y_train[int(len(y_train)*0.3):])

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

BATCH_SIZE = 2
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


# On définis notre modèle
model = tf.keras.Sequential([
        layers.Conv2D(4, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(8, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(4, activation='linear')
    ])

model.compile(optimizer='adam', loss='mean_absolute_error')

# Entrainement
epochs=30
history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs)
model.summary()

model.save("saved_model/face_extractor")









